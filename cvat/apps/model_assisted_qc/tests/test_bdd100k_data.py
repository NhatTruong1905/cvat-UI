from django.test import SimpleTestCase

from cvat.apps.model_assisted_qc.phase1_dataset_ground_truth.bdd100k import (
    corrupt_annotations,
    load_bdd100k_records,
    map_bdd100k_class,
    split_by_sequence,
    stratified_sample,
)


def _row(name, video, weather="clear", category="car"):
    return {
        "name": name,
        "videoName": video,
        "attributes": {"weather": weather, "timeofday": "daytime", "scene": "city street"},
        "labels": [
            {"id": 1, "category": category, "box2d": {"x1": 1, "y1": 2, "x2": 20, "y2": 30}}
        ],
    }


class BDD100KDataTest(SimpleTestCase):
    def setUp(self):
        self.records = load_bdd100k_records(
            [
                _row("a-1.jpg", "a", "clear"),
                _row("a-2.jpg", "a", "rainy"),
                _row("b-1.jpg", "b", "snowy", "motorcycle"),
                _row("c-1.jpg", "c", "clear", "bicycle"),
            ]
        )

    def test_loads_normalized_records_and_maps_taxonomy(self):
        self.assertEqual(self.records[0]["annotations"][0]["bbox_xyxy"], [1.0, 2.0, 20.0, 30.0])
        self.assertEqual(map_bdd100k_class("motorcycle"), "motorcycle")
        self.assertEqual(map_bdd100k_class("other vehicle"), "UNMAPPED")

    def test_split_is_deterministic_and_has_no_sequence_overlap(self):
        first = split_by_sequence(self.records, seed=7)
        second = split_by_sequence(self.records, seed=7)

        self.assertEqual(first, second)
        calibration_groups = {record["source_video_id"] for record in first[0]}
        held_out_groups = {record["source_video_id"] for record in first[1]}
        self.assertFalse(calibration_groups & held_out_groups)

    def test_stratified_sample_is_deterministic(self):
        first = stratified_sample(self.records, 3, seed=9)
        second = stratified_sample(self.records, 3, seed=9)

        self.assertEqual(first, second)
        self.assertEqual(len({record["weather"] for record in first}), 3)

    def test_corruption_copies_input_and_records_before_after(self):
        corrupted, manifest = corrupt_annotations(self.records, 2, seed=5)

        self.assertEqual(sum(len(row["annotations"]) for row in self.records), 4)
        self.assertEqual({item["error_type"] for item in manifest}, {"MISSING_OBJECT", "WRONG_CLASS"})
        self.assertTrue(all(item["error_id"] for item in manifest))
        self.assertNotEqual(corrupted, self.records)
