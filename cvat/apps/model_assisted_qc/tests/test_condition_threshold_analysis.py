from django.test import SimpleTestCase

from types import SimpleNamespace

from cvat.apps.model_assisted_qc.phase3_condition_threshold import (
    run_threshold_analysis,
    summarize_by_condition,
)


class ConditionThresholdAnalysisTest(SimpleTestCase):
    def test_runs_grid_and_selects_best_objective(self):
        calls = []

        def runner(**options):
            calls.append(options)
            confidence = options["confidence"]
            iou = options["iou_threshold"]
            return {
                "map50": confidence + iou,
                "map50_95": confidence,
                "precision": confidence,
                "recall": iou,
                "evaluated_images": 5,
            }, []

        results, best = run_threshold_analysis(
            ground_truth_job_id=2,
            model_path="/models/model.pt",
            confidence_values=[0.2, 0.4],
            iou_values=[0.5, 0.7],
            objective="map50",
            conditions={"weather": "rain"},
            evaluation_runner=runner,
        )

        self.assertEqual(len(results), 4)
        self.assertEqual(len(calls), 4)
        self.assertFalse(calls[0]["include_predictions"])
        self.assertEqual(best["confidence"], 0.4)
        self.assertEqual(best["iou_threshold"], 0.7)
        self.assertEqual(best["conditions"], {"weather": "rain"})

    def test_f1_objective_handles_zero_values(self):
        def runner(**options):
            confidence = options["confidence"]
            return {
                "map50": 0,
                "map50_95": 0,
                "precision": confidence,
                "recall": 1 - confidence,
                "evaluated_images": 1,
            }, []

        _, best = run_threshold_analysis(
            1,
            "model.pt",
            [0.1, 0.5],
            [0.5],
            "f1",
            {},
            runner,
        )

        self.assertEqual(best["confidence"], 0.5)
        self.assertEqual(best["f1"], 0.5)

    def test_summarizes_best_results_by_condition(self):
        analyses = [
            SimpleNamespace(
                conditions={"weather": "rain"},
                best_result={"map50": 0.6, "map50_95": 0.4, "precision": 0.7, "recall": 0.5, "f1": 0.58},
            ),
            SimpleNamespace(
                conditions={"weather": "rain"},
                best_result={"map50": 0.8, "map50_95": 0.6, "precision": 0.9, "recall": 0.7, "f1": 0.78},
            ),
            SimpleNamespace(
                conditions={"weather": "clear"},
                best_result={"map50": 0.9, "map50_95": 0.8, "precision": 0.9, "recall": 0.8, "f1": 0.85},
            ),
        ]

        summaries = summarize_by_condition(analyses)

        self.assertEqual(summaries[0]["conditions"], {"weather": "clear"})
        self.assertEqual(summaries[1]["runs"], 2)
        self.assertAlmostEqual(summaries[1]["map50_95"], 0.5)
