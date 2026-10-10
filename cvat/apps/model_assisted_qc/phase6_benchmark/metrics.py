import random
from collections import Counter


def _identity(item):
    return item["type"], int(item["frame"])


def _evaluate(alerts, verified_errors, top_k):
    truth = Counter(_identity(error) for error in verified_errors)
    found = Counter(_identity(alert) for alert in alerts)
    true_positives = sum((truth & found).values())
    precision = true_positives / len(alerts) if alerts else 0.0
    verified_count = sum(truth.values())
    recall = true_positives / verified_count if verified_count else 0.0
    precision_at_k = {}
    for value in top_k:
        selected = alerts[: min(value, len(alerts))]
        denominator = len(selected)
        precision_at_k[str(value)] = (
            sum((Counter(_identity(alert) for alert in selected) & truth).values()) / denominator
            if denominator
            else 0.0
        )
    return {
        "alerts": len(alerts),
        "verified_errors": verified_count,
        "true_positives": true_positives,
        "qc_precision": precision,
        "qc_recall": recall,
        "precision_at_k": precision_at_k,
    }


def benchmark_alerts(alerts, verified_errors, top_k=(20, 50), seed=42):
    ranked = sorted(alerts, key=lambda item: (-item["risk_score"], item["frame"], item["id"]))
    unranked = sorted(alerts, key=lambda item: item["id"])
    random_order = list(unranked)
    random.Random(seed).shuffle(random_order)
    report = {
        "seed": seed,
        "top_k": list(top_k),
        "baselines": {
            "random": _evaluate(random_order, verified_errors, top_k),
            "unranked": _evaluate(unranked, verified_errors, top_k),
            "risk_ranked": _evaluate(ranked, verified_errors, top_k),
        },
        "by_type": {},
    }
    for alert_type in ("POSSIBLE_MISSING_OBJECT", "POSSIBLE_WRONG_CLASS"):
        report["by_type"][alert_type] = _evaluate(
            [item for item in ranked if item["type"] == alert_type],
            [item for item in verified_errors if item["type"] == alert_type],
            top_k,
        )
    return report
