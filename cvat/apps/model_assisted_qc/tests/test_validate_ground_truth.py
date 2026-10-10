from django.test import SimpleTestCase

from cvat.apps.model_assisted_qc.phase1_dataset_ground_truth.validate_ground_truth import (
    validate_ground_truth,
)


class GroundTruthValidationTest(SimpleTestCase):
    labels = {1: "car", 2: "rider", 3: "bicycle", 4: "custom"}
    image_sizes = {0: (100, 100), 1: (100, 100)}

    def validate(self, shapes, frames=(0,)):
        return validate_ground_truth(
            {"shapes": shapes, "tracks": []},
            self.labels,
            self.image_sizes,
            frames,
        )

    @staticmethod
    def shape(shape_id, label_id, points, frame=0):
        return {
            "id": shape_id,
            "label_id": label_id,
            "type": "rectangle",
            "frame": frame,
            "outside": False,
            "points": points,
        }

    def test_valid_annotations_pass(self):
        result = self.validate([self.shape(1, 1, [10, 10, 50, 50])])

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["summary"]["boxes_checked"], 1)

    def test_invalid_label_and_bounds_block(self):
        result = self.validate([self.shape(1, 4, [-1, 10, 110, 50])])

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["summary"]["blockers"], 2)

    def test_small_duplicate_and_empty_frame_warn(self):
        shapes = [
            self.shape(1, 1, [10, 10, 20, 20]),
            self.shape(2, 1, [10, 10, 20, 20]),
        ]
        result = self.validate(shapes, frames=(0, 1))

        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["summary"]["rules"]["small_box"], 2)
        self.assertEqual(result["summary"]["rules"]["duplicate_box"], 1)
        self.assertEqual(result["summary"]["rules"]["empty_frame"], 1)

    def test_rider_requires_nearby_vehicle(self):
        result = self.validate([self.shape(1, 2, [10, 10, 40, 60])])

        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["summary"]["rules"]["rider_association"], 1)
