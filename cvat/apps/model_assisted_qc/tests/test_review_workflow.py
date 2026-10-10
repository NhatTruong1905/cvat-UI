from django.test import SimpleTestCase

from cvat.apps.model_assisted_qc.models import QCAlert
from cvat.apps.model_assisted_qc.phase5_human_review.workflow import next_review_status


class ReviewWorkflowTest(SimpleTestCase):
    def test_pending_can_be_confirmed_or_rejected(self):
        self.assertEqual(
            next_review_status(QCAlert.Status.PENDING, "CONFIRMED"),
            QCAlert.Status.CONFIRMED,
        )
        self.assertEqual(
            next_review_status(QCAlert.Status.PENDING, "REJECTED"),
            QCAlert.Status.REJECTED,
        )

    def test_only_confirmed_can_be_resolved(self):
        with self.assertRaises(ValueError):
            next_review_status(QCAlert.Status.PENDING, "RESOLVED")
        self.assertEqual(
            next_review_status(QCAlert.Status.CONFIRMED, "RESOLVED"),
            QCAlert.Status.RESOLVED,
        )

    def test_terminal_states_cannot_be_changed(self):
        with self.assertRaises(ValueError):
            next_review_status(QCAlert.Status.REJECTED, "CONFIRMED")
        with self.assertRaises(ValueError):
            next_review_status(QCAlert.Status.RESOLVED, "REJECTED")
