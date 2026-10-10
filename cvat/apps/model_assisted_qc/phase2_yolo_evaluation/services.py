import gc
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml
from PIL import Image

from cvat.apps.dataset_manager.views import export_job_as_dataset


YOLO_FORMAT = "Ultralytics YOLO Detection 1.0"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inference_provenance(ground_truth_job_id, model_path, confidence, iou_threshold):
    model_file = resolve_evaluation_path(model_path)
    provenance = {
        "prediction_schema_version": 2,
        "ground_truth_job_id": ground_truth_job_id,
        "model_path": str(model_file),
        "model_sha256": _sha256(model_file),
        "inference_confidence": confidence,
        "nms_iou": iou_threshold,
        "backend": "ultralytics",
    }
    encoded = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    return provenance, hashlib.sha256(encoded).hexdigest()


def resolve_evaluation_path(value):
    root = Path(os.getenv("CVAT_YOLO_EVALUATION_ROOT", "/opt/cvat/evaluation-data")).resolve()
    path = Path(value).resolve(strict=True)
    if path != root and root not in path.parents:
        raise ValueError(f"Evaluation files must be inside {root}")
    return path


def find_dataset_yaml(dataset_root):
    candidates = list(dataset_root.rglob("data.yaml")) + list(dataset_root.rglob("dataset.yaml"))
    if not candidates:
        raise RuntimeError("The exported Ground Truth dataset does not contain a YOLO YAML file")
    return candidates[0]


def ensure_validation_split(data_file):
    with data_file.open(encoding="utf-8") as stream:
        configuration = yaml.safe_load(stream) or {}
    image_directory = data_file.parent / "images" / "train"
    if not image_directory.exists():
        raise RuntimeError("The exported Ground Truth dataset has no images/train directory")
    configuration["path"] = str(data_file.parent.resolve())
    configuration["train"] = "images/train"
    configuration["val"] = "images/train"
    with data_file.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(configuration, stream, sort_keys=False)


def release_ultralytics_runtime(model):
    model.predictor = None
    model.metrics = None
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def read_validation_predictions(image_files, labels_directory, class_names, provenance):
    predictions = []
    for image_file in image_files:
        boxes = []
        prediction_file = labels_directory / f"{image_file.stem}.txt"
        if prediction_file.exists():
            with Image.open(image_file) as image:
                image_width, image_height = image.size
            for line in prediction_file.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) != 6:
                    continue
                class_id, center_x, center_y, width, height, score = map(float, values)
                class_id = int(class_id)
                center_x *= image_width
                center_y *= image_height
                width *= image_width
                height *= image_height
                boxes.append(
                    {
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "confidence": score,
                        "xyxy": [
                            center_x - width / 2,
                            center_y - height / 2,
                            center_x + width / 2,
                            center_y + height / 2,
                        ],
                    }
                )
        predictions.append(
            {
                "image": image_file.name,
                "provenance": provenance,
                "boxes": boxes,
            }
        )
    return predictions


def run_ultralytics(
    ground_truth_job_id, model_path, confidence, iou_threshold, include_predictions=True
):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the optional 'ultralytics' package in the CVAT backend") from exc

    model_file = resolve_evaluation_path(model_path)
    archive_path = export_job_as_dataset(ground_truth_job_id, YOLO_FORMAT)
    with tempfile.TemporaryDirectory(prefix="cvat-yolo-evaluation-") as temp_dir:
        dataset_root = Path(temp_dir)
        shutil.unpack_archive(archive_path, dataset_root, "zip")
        data_file = find_dataset_yaml(dataset_root)
        ensure_validation_split(data_file)
        image_files = [
            path
            for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
            for path in dataset_root.rglob(extension)
        ]
        if not image_files:
            raise RuntimeError("The exported Ground Truth dataset does not contain images")
        model = YOLO(str(model_file))
        validation = model.val(
            data=str(data_file),
            conf=confidence,
            iou=iou_threshold,
            batch=1,
            workers=0,
            plots=False,
            save_txt=include_predictions,
            save_conf=include_predictions,
            project=str(dataset_root / "runs"),
            name="validation",
            exist_ok=True,
        )
        confusion = (
            validation.confusion_matrix.matrix.tolist() if validation.confusion_matrix else []
        )
        box = validation.box
        metrics = {
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "confusion_matrix": confusion,
            "evaluated_images": len(image_files),
        }
        predictions = []
        if include_predictions:
            predictions = read_validation_predictions(
                image_files=image_files,
                labels_directory=Path(validation.save_dir) / "labels",
                class_names=model.names,
                provenance={
                    "model_path": str(model_file),
                    "inference_confidence": confidence,
                    "nms_iou": iou_threshold,
                },
            )
        del box
        del validation
        release_ultralytics_runtime(model)
        return metrics, predictions
