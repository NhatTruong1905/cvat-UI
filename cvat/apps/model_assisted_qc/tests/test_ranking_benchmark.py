from django.test import SimpleTestCase

from cvat.apps.model_assisted_qc.phase4_risk_review import rank_alerts, score_alert
from cvat.apps.model_assisted_qc.phase6_benchmark import benchmark_alerts


def _alert(alert_id, frame, alert_type, confidence, risk_score=0):
    return {
        "id": alert_id,
        "frame": frame,
        "type": alert_type,
        "prediction": {"confidence": confidence, "bbox_xyxy": [0, 0, 10, 10]},
        "annotation": None,
        "iou": None,
        "risk_score": risk_score,
        "reason_codes": ["UNMATCHED_PREDICTION"],
    }


class RankingBenchmarkTest(SimpleTestCase):
    def test_score_has_transparent_breakdown(self):
        score, breakdown = score_alert(
            _alert(1, 0, "POSSIBLE_MISSING_OBJECT", 0.8), condition_reliability=0.5
        )

        self.assertAlmostEqual(score, 0.725)
        self.assertAlmostEqual(score, sum(breakdown.values()))

    def test_ranking_is_deterministic(self):
        alerts = [
            _alert(2, 2, "POSSIBLE_MISSING_OBJECT", 0.5),
            _alert(1, 1, "POSSIBLE_MISSING_OBJECT", 0.9),
        ]

        self.assertEqual(rank_alerts(alerts), rank_alerts(alerts))
        self.assertEqual(rank_alerts(alerts)[0]["frame"], 1)

    def test_unmatched_human_box_has_reviewable_risk(self):
        alert = _alert(1, 0, "POSSIBLE_EXTRA_OBJECT", 0)

        score, breakdown = score_alert(alert, condition_reliability=1)

        self.assertEqual(score, 0.7)
        self.assertEqual(breakdown["model_disagreement"], 0.45)

    def test_benchmark_handles_top_k_duplicates_and_zero_denominators(self):
        alerts = [
            {**_alert(1, 1, "POSSIBLE_MISSING_OBJECT", 0.9), "risk_score": 0.9},
            {**_alert(2, 2, "POSSIBLE_WRONG_CLASS", 0.8), "risk_score": 0.8},
            {**_alert(3, 3, "POSSIBLE_MISSING_OBJECT", 0.7), "risk_score": 0.7},
        ]
        verified = [
            {"error_id": "a", "type": "POSSIBLE_MISSING_OBJECT", "frame": 1},
            {"error_id": "b", "type": "POSSIBLE_WRONG_CLASS", "frame": 2},
        ]

        report = benchmark_alerts(alerts, verified, top_k=[1, 10], seed=7)

        ranked = report["baselines"]["risk_ranked"]
        self.assertEqual(ranked["qc_recall"], 1)
        self.assertEqual(ranked["precision_at_k"]["1"], 1)
        self.assertAlmostEqual(ranked["precision_at_k"]["10"], 2 / 3)

    def test_empty_alerts_do_not_divide_by_zero(self):
        report = benchmark_alerts(
            [],
            [{"error_id": "a", "type": "POSSIBLE_MISSING_OBJECT", "frame": 1}],
            top_k=[20],
        )

        self.assertEqual(report["baselines"]["risk_ranked"]["qc_precision"], 0)
        self.assertEqual(report["baselines"]["risk_ranked"]["qc_recall"], 0)
