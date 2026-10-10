from itertools import product
import json


def _score(result, objective):
    if objective == "f1":
        precision = result.get("precision", 0)
        recall = result.get("recall", 0)
        return 2 * precision * recall / (precision + recall) if precision + recall else 0
    return result.get(objective, 0)


def run_threshold_analysis(
    ground_truth_job_id,
    model_path,
    confidence_values,
    iou_values,
    objective,
    conditions,
    evaluation_runner,
):
    results = []
    for confidence, iou_threshold in product(confidence_values, iou_values):
        metrics, _ = evaluation_runner(
            ground_truth_job_id=ground_truth_job_id,
            model_path=model_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            include_predictions=False,
        )
        result = {
            "confidence": confidence,
            "iou_threshold": iou_threshold,
            "map50": metrics.get("map50", 0),
            "map50_95": metrics.get("map50_95", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "evaluated_images": metrics.get("evaluated_images", 0),
            "conditions": conditions,
        }
        result["f1"] = _score(result, "f1")
        results.append(result)
    best_result = max(
        results,
        key=lambda result: (_score(result, objective), result["map50_95"], result["map50"]),
    )
    return results, best_result


def summarize_by_condition(analyses):
    groups = {}
    for analysis in analyses:
        if not analysis.best_result:
            continue
        condition_key = json.dumps(analysis.conditions or {}, sort_keys=True)
        group = groups.setdefault(
            condition_key,
            {
                "conditions": analysis.conditions or {},
                "runs": 0,
                "map50": 0.0,
                "map50_95": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        )
        group["runs"] += 1
        for metric in ("map50", "map50_95", "precision", "recall", "f1"):
            group[metric] += float(analysis.best_result.get(metric, 0))
    summaries = []
    for group in groups.values():
        runs = group["runs"]
        for metric in ("map50", "map50_95", "precision", "recall", "f1"):
            group[metric] /= runs
        summaries.append(group)
    return sorted(summaries, key=lambda group: group["map50_95"], reverse=True)
