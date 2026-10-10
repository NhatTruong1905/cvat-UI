def score_alert(alert, condition_reliability=1.0):
    confidence = float(alert["prediction"].get("confidence", 0))
    reliability = min(1.0, max(0.0, float(condition_reliability)))
    if alert["type"] == "POSSIBLE_WRONG_CLASS":
        iou = float(alert.get("iou") or 0)
        breakdown = {
            "confidence": 0.45 * confidence,
            "match_iou": 0.35 * iou,
            "condition_reliability": 0.20 * reliability,
        }
    elif alert["type"] == "POSSIBLE_EXTRA_OBJECT":
        breakdown = {
            "model_disagreement": 0.45,
            "condition_reliability": 0.25 * reliability,
        }
    else:
        breakdown = {
            "confidence": 0.75 * confidence,
            "condition_reliability": 0.25 * reliability,
        }
    return round(sum(breakdown.values()), 6), {
        key: round(value, 6) for key, value in breakdown.items()
    }


def rank_alerts(alerts, condition_reliability=1.0):
    ranked = []
    for alert in alerts:
        risk_score, score_breakdown = score_alert(alert, condition_reliability)
        ranked.append({**alert, "risk_score": risk_score, "score_breakdown": score_breakdown})
    return sorted(
        ranked,
        key=lambda alert: (
            -alert["risk_score"],
            alert.get("frame", 0),
            alert["type"],
            str(alert["prediction"].get("bbox_xyxy", [])),
        ),
    )
