from django.test import SimpleTestCase

from cvat.apps.model_assisted_qc.phase3_condition_threshold.qc_detection import (
    box_iou,
    detect_qc_errors,
    match_boxes,
)
from cvat.apps.model_assisted_qc.phase4_risk_review.services import normalize_prediction_class
from cvat.apps.model_assisted_qc.phase4_risk_review.serializers import (
    AlertFilterSerializer,
    QueueCreateSerializer,
)


def _prediction(class_name="car", box=(0, 0, 20, 20), confidence=0.9):
    return {"class_name": class_name, "bbox_xyxy": list(box), "confidence": confidence}


def _annotation(class_name="car", box=(0, 0, 20, 20)):
    return {"class_name": class_name, "bbox_xyxy": list(box), "annotation_id": 1}


class QCDetectionTest(SimpleTestCase):
    def test_queue_requires_current_annotation_job(self):
        serializer = QueueCreateSerializer(data={"evaluation_run_id": 1})

        self.assertFalse(serializer.is_valid())
        self.assertIn("job_id", serializer.errors)

    def test_alerts_can_be_filtered_by_annotation_job(self):
        serializer = AlertFilterSerializer(data={"job_id": 7})

        self.assertTrue(serializer.is_valid())
        self.assertEqual(serializer.validated_data["job_id"], 7)

    def test_coco_person_maps_to_bdd100k_pedestrian(self):
        prediction = _prediction(normalize_prediction_class("person"))

        self.assertEqual(detect_qc_errors([prediction], [_annotation("pedestrian")]), [])

    def test_iou_rejects_invalid_and_computes_overlap(self):
        self.assertEqual(box_iou((0, 0, 0, 10), (0, 0, 10, 10)), 0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_same_class_match_has_no_alert(self):
        self.assertEqual(detect_qc_errors([_prediction()], [_annotation()]), [])

    def test_wrong_class_is_not_also_missing(self):
        alerts = detect_qc_errors([_prediction("truck")], [_annotation("car")])

        self.assertEqual([alert["type"] for alert in alerts], ["POSSIBLE_WRONG_CLASS"])

    def test_unmatched_prediction_is_missing_candidate(self):
        alerts = detect_qc_errors([_prediction()], [])

        self.assertEqual(alerts[0]["type"], "POSSIBLE_MISSING_OBJECT")

    def test_unmatched_annotation_creates_human_box_alert(self):
        alerts = detect_qc_errors([], [_annotation()])

        self.assertEqual(alerts[0]["type"], "POSSIBLE_EXTRA_OBJECT")
        self.assertEqual(alerts[0]["annotation"]["annotation_id"], 1)

    def test_duplicate_prediction_is_suppressed(self):
        predictions = [_prediction(confidence=0.9), _prediction(confidence=0.8)]

        self.assertEqual(detect_qc_errors(predictions, [_annotation()]), [])

    def test_low_iou_different_class_is_missing_not_wrong_class(self):
        alerts = detect_qc_errors(
            [_prediction("truck", (50, 50, 70, 70))],
            [_annotation("car", (0, 0, 20, 20))],
        )

        self.assertEqual(
            [alert["type"] for alert in alerts],
            ["POSSIBLE_MISSING_OBJECT", "POSSIBLE_EXTRA_OBJECT"],
        )

    def test_unmapped_invalid_and_low_confidence_predictions_are_ignored(self):
        predictions = [
            _prediction("UNMAPPED"),
            _prediction(box=(0, 0, 0, 20)),
            _prediction(confidence=0.1),
        ]

        self.assertEqual(detect_qc_errors(predictions, []), [])

    def test_hungarian_maximizes_valid_matches(self):
        predictions = [_prediction(box=(0, 0, 10, 10)), _prediction(box=(8, 0, 18, 10))]
        annotations = [_annotation(box=(0, 0, 10, 10)), _annotation(box=(8, 0, 18, 10))]

        matches, unmatched_predictions, unmatched_annotations = match_boxes(
            predictions, annotations, match_iou=0.5
        )

        self.assertEqual(len(matches), 2)
        self.assertEqual(unmatched_predictions, [])
        self.assertEqual(unmatched_annotations, [])
