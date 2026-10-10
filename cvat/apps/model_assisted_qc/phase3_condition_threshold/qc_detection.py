from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Match:
    prediction_index: int
    annotation_index: int
    iou: float


def box_iou(first, second):
    if not _valid_box(first) or not _valid_box(second):
        return 0.0
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _valid_box(box):
    return len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def deduplicate_predictions(predictions, iou_threshold=0.9):
    kept = []
    for prediction in sorted(predictions, key=lambda item: item.get("confidence", 0), reverse=True):
        if not _valid_box(prediction.get("bbox_xyxy", [])):
            continue
        duplicate = any(
            prediction.get("class_name") == existing.get("class_name")
            and box_iou(prediction["bbox_xyxy"], existing["bbox_xyxy"]) >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(prediction)
    return kept


def match_boxes(predictions, annotations, match_iou=0.5):
    prediction_count = len(predictions)
    annotation_count = len(annotations)
    if not prediction_count or not annotation_count:
        return [], list(range(prediction_count)), list(range(annotation_count))
    size = prediction_count + annotation_count
    costs = np.zeros((size, size), dtype=float)
    costs[:prediction_count, :annotation_count] = 1_000_000
    ious = np.zeros((prediction_count, annotation_count), dtype=float)
    for prediction_index, prediction in enumerate(predictions):
        for annotation_index, annotation in enumerate(annotations):
            iou = box_iou(prediction["bbox_xyxy"], annotation["bbox_xyxy"])
            ious[prediction_index, annotation_index] = iou
            if iou >= match_iou:
                costs[prediction_index, annotation_index] = -1_000 - iou
    rows, columns = linear_sum_assignment(costs)
    matches = []
    for row, column in zip(rows, columns):
        if row < prediction_count and column < annotation_count and ious[row, column] >= match_iou:
            matches.append(Match(row, column, float(ious[row, column])))
    matched_predictions = {match.prediction_index for match in matches}
    matched_annotations = {match.annotation_index for match in matches}
    return (
        matches,
        [index for index in range(prediction_count) if index not in matched_predictions],
        [index for index in range(annotation_count) if index not in matched_annotations],
    )


def detect_qc_errors(
    predictions,
    annotations,
    match_iou=0.5,
    min_confidence=0.5,
    dedup_iou=0.9,
):
    valid_annotations = [
        annotation
        for annotation in annotations
        if _valid_box(annotation.get("bbox_xyxy", []))
        and annotation.get("class_name") not in {None, "UNMAPPED"}
    ]
    eligible_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("confidence", 0) >= min_confidence
        and prediction.get("class_name") not in {None, "UNMAPPED"}
    ]
    eligible_predictions = deduplicate_predictions(eligible_predictions, dedup_iou)
    matches, unmatched_predictions, unmatched_annotations = match_boxes(
        eligible_predictions, valid_annotations, match_iou
    )
    alerts = []
    for match in matches:
        prediction = eligible_predictions[match.prediction_index]
        annotation = valid_annotations[match.annotation_index]
        if prediction["class_name"] != annotation["class_name"]:
            alerts.append(
                {
                    "type": "POSSIBLE_WRONG_CLASS",
                    "prediction": prediction,
                    "annotation": annotation,
                    "iou": match.iou,
                    "reason_codes": ["CLASS_MISMATCH"],
                }
            )
    for prediction_index in unmatched_predictions:
        prediction = eligible_predictions[prediction_index]
        alerts.append(
            {
                "type": "POSSIBLE_MISSING_OBJECT",
                "prediction": prediction,
                "annotation": None,
                "iou": None,
                "reason_codes": ["UNMATCHED_PREDICTION"],
            }
        )
    for annotation_index in unmatched_annotations:
        annotation = valid_annotations[annotation_index]
        alerts.append(
            {
                "type": "POSSIBLE_EXTRA_OBJECT",
                "prediction": {
                    "class_id": None,
                    "class_name": "NO_MATCH",
                    "confidence": 0.0,
                    "bbox_xyxy": annotation["bbox_xyxy"],
                },
                "annotation": annotation,
                "iou": None,
                "reason_codes": ["UNMATCHED_ANNOTATION"],
            }
        )
    return alerts
