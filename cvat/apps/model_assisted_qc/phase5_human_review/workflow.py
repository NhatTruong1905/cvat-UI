from ..models import QCAlert


TRANSITIONS = {
    QCAlert.Status.PENDING: {
        "START_REVIEW": QCAlert.Status.IN_REVIEW,
        "CONFIRMED": QCAlert.Status.CONFIRMED,
        "REJECTED": QCAlert.Status.REJECTED,
        "NEEDS_MORE_INFO": QCAlert.Status.IN_REVIEW,
    },
    QCAlert.Status.IN_REVIEW: {
        "CONFIRMED": QCAlert.Status.CONFIRMED,
        "REJECTED": QCAlert.Status.REJECTED,
        "NEEDS_MORE_INFO": QCAlert.Status.IN_REVIEW,
    },
    QCAlert.Status.CONFIRMED: {"RESOLVED": QCAlert.Status.RESOLVED},
    QCAlert.Status.REJECTED: {},
    QCAlert.Status.RESOLVED: {},
}


def next_review_status(current_status, decision):
    try:
        return TRANSITIONS[current_status][decision]
    except KeyError as exc:
        raise ValueError(f"Decision {decision} is not valid from status {current_status}") from exc
