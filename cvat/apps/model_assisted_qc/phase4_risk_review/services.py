import hashlib
import json
from collections import defaultdict
from pathlib import Path

from cvat.apps.dataset_manager.task import get_job_data

from ..models import QCAlert
from ..phase3_condition_threshold.qc_detection import detect_qc_errors
from .ranking import rank_alerts


BDD100K_CLASS_ALIASES = {
    "person": "pedestrian",
}


def normalize_prediction_class(class_name):
    normalized = str(class_name).strip().lower()
    return BDD100K_CLASS_ALIASES.get(normalized, normalized)


def _stable_key(source_job_id, frame, alert):
    payload = {
        "source_job_id": source_job_id,
        "frame": frame,
        "type": alert["type"],
        "prediction": alert["prediction"],
        "annotation": alert["annotation"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _annotations_by_frame(annotation_data, labels):
    result = defaultdict(list)
    for shape in annotation_data.get("shapes", []):
        if shape.get("outside") or shape.get("type") != "rectangle":
            continue
        result[int(shape["frame"])].append(
            {
                "annotation_id": shape.get("id"),
                "class_name": labels.get(shape.get("label_id"), "UNMAPPED"),
                "bbox_xyxy": [float(value) for value in shape.get("points", [])],
            }
        )
    for track in annotation_data.get("tracks", []):
        class_name = labels.get(track.get("label_id"), "UNMAPPED")
        for shape in track.get("shapes", []):
            if shape.get("outside") or shape.get("type") != "rectangle":
                continue
            result[int(shape["frame"])].append(
                {
                    "annotation_id": shape.get("id"),
                    "track_id": track.get("id"),
                    "class_name": class_name,
                    "bbox_xyxy": [float(value) for value in shape.get("points", [])],
                }
            )
    return result


def generate_alert_queue(evaluation_run, source_job, options):
    task = evaluation_run.task
    labels = {label.id: label.name for label in task.get_labels()}
    annotations = _annotations_by_frame(
        get_job_data(source_job.id), labels
    )
    image_frames = {
        Path(image.path).name: image.frame
        for image in task.data.images.filter(
            frame__in=evaluation_run.ground_truth_job.segment.frame_set
        )
    }
    candidates = []
    for image_prediction in evaluation_run.predictions:
        image_name = Path(image_prediction["image"]).name
        frame = image_frames.get(image_name)
        if frame is None:
            continue
        predictions = [
            {
                "class_id": item.get("class_id"),
                "class_name": normalize_prediction_class(item.get("class_name", "UNMAPPED")),
                "confidence": item["confidence"],
                "bbox_xyxy": item["xyxy"],
            }
            for item in image_prediction.get("boxes", [])
        ]
        frame_alerts = detect_qc_errors(
            predictions,
            annotations.get(frame, []),
            match_iou=options["match_iou"],
            min_confidence=options["min_confidence"],
            dedup_iou=options["dedup_iou"],
        )
        candidates.extend({**alert, "frame": frame, "image_name": image_name} for alert in frame_alerts)
    ranked = rank_alerts(candidates, options["condition_reliability"])
    active_keys = {_stable_key(source_job.id, alert["frame"], alert) for alert in ranked}
    stale_alerts = QCAlert.objects.filter(
        evaluation_run=evaluation_run,
        source_job=source_job,
        status__in=(QCAlert.Status.PENDING, QCAlert.Status.IN_REVIEW),
        decisions__isnull=True,
    )
    if active_keys:
        stale_alerts = stale_alerts.exclude(stable_key__in=active_keys)
    stale_alerts.delete()
    created = 0
    for alert in ranked:
        _, was_created = QCAlert.objects.update_or_create(
            evaluation_run=evaluation_run,
            stable_key=_stable_key(source_job.id, alert["frame"], alert),
            defaults={
                "task": task,
                "source_job": source_job,
                "frame": alert["frame"],
                "image_name": alert["image_name"],
                "type": alert["type"],
                "prediction": alert["prediction"],
                "annotation": alert["annotation"],
                "iou": alert["iou"],
                "risk_score": alert["risk_score"],
                "score_breakdown": alert["score_breakdown"],
                "reason_codes": alert["reason_codes"],
            },
        )
        created += int(was_created)
    return {"total": len(ranked), "created": created, "reused": len(ranked) - created}
