import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from cvat.apps.model_assisted_qc.phase2_yolo_evaluation.services import (
    build_inference_provenance,
    read_validation_predictions,
    release_ultralytics_runtime,
)
from PIL import Image


class InferenceProvenanceTest(SimpleTestCase):
    def test_reads_predictions_written_by_ultralytics_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_file = root / "frame.jpg"
            labels_directory = root / "labels"
            labels_directory.mkdir()
            Image.new("RGB", (200, 100)).save(image_file)
            (labels_directory / "frame.txt").write_text(
                "2 0.5 0.5 0.2 0.4 0.75\n", encoding="utf-8"
            )

            predictions = read_validation_predictions(
                [image_file], labels_directory, {2: "car"}, {"model_path": "model.pt"}
            )

        self.assertEqual(predictions[0]["boxes"][0]["class_name"], "car")
        self.assertEqual(predictions[0]["boxes"][0]["xyxy"], [80.0, 30.0, 120.0, 70.0])
        self.assertEqual(predictions[0]["boxes"][0]["confidence"], 0.75)

    def test_release_ultralytics_runtime_drops_cached_objects(self):
        class FakeModel:
            predictor = object()
            metrics = object()

        model = FakeModel()
        release_ultralytics_runtime(model)

        self.assertIsNone(model.predictor)
        self.assertIsNone(model.metrics)

    def test_same_model_and_config_produce_same_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.pt"
            model.write_bytes(b"weights-v1")
            with patch.dict(os.environ, {"CVAT_YOLO_EVALUATION_ROOT": directory}):
                first = build_inference_provenance(7, str(model), 0.25, 0.7)
                second = build_inference_provenance(7, str(model), 0.25, 0.7)

        self.assertEqual(first, second)
        self.assertEqual(first[0]["backend"], "ultralytics")

    def test_model_or_configuration_changes_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.pt"
            model.write_bytes(b"weights-v1")
            with patch.dict(os.environ, {"CVAT_YOLO_EVALUATION_ROOT": directory}):
                first_key = build_inference_provenance(7, str(model), 0.25, 0.7)[1]
                changed_config_key = build_inference_provenance(7, str(model), 0.5, 0.7)[1]
                model.write_bytes(b"weights-v2")
                changed_model_key = build_inference_provenance(7, str(model), 0.25, 0.7)[1]

        self.assertNotEqual(first_key, changed_config_key)
        self.assertNotEqual(first_key, changed_model_key)
