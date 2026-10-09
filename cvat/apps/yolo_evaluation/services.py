import os
import shutil
import tempfile
from pathlib import Path

import yaml

from cvat.apps.dataset_manager.views import export_job_as_dataset


YOLO_FORMAT = "Ultralytics YOLO Detection 1.0"


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


def run_ultralytics(ground_truth_job_id, model_path, confidence, iou_threshold):
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
            plots=False,
            project=str(dataset_root / "runs"),
            name="validation",
            exist_ok=True,
        )
        confusion = (
            validation.confusion_matrix.matrix.tolist() if validation.confusion_matrix else []
        )
        predictions = []
        for result in model.predict(
            source=[str(path) for path in image_files],
            conf=confidence,
            iou=iou_threshold,
            stream=True,
            project=str(dataset_root / "runs"),
            name="prediction",
            exist_ok=True,
            save=False,
        ):
            boxes = result.boxes
            predictions.append(
                {
                    "image": Path(result.path).name,
                    "boxes": []
                    if boxes is None
                    else [
                        {
                            "class_id": int(class_id),
                            "confidence": float(score),
                            "xyxy": [float(value) for value in coordinates],
                        }
                        for class_id, score, coordinates in zip(
                            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
                        )
                    ],
                }
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
        return metrics, predictions
