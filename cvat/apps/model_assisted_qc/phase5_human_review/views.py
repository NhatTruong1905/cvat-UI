from django.db import transaction
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..common import get_task
from ..models import QCAlert, ReviewDecision
from ..phase4_risk_review.views import serialize_alert
from .serializers import ReviewDecisionSerializer
from .workflow import next_review_status


class AlertReviewView(APIView):
    permission_classes = (IsAuthenticated,)

    @transaction.atomic
    def post(self, request, alert_id):
        serializer = ReviewDecisionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        alert = QCAlert.objects.select_for_update().filter(id=alert_id).first()
        if alert is None:
            return Response({"error": "Alert was not found"}, status=status.HTTP_404_NOT_FOUND)
        get_task(request, alert.task_id)
        decision = serializer.validated_data["decision"]
        try:
            resulting_status = next_review_status(alert.status, decision)
        except ValueError as exc:
            return Response(
                {"error": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        previous_status = alert.status
        alert.status = resulting_status
        alert.save(update_fields=["status", "updated_date"])
        ReviewDecision.objects.create(
            alert=alert,
            reviewer=request.user,
            decision=decision,
            previous_status=previous_status,
            resulting_status=alert.status,
            notes=serializer.validated_data["notes"],
        )
        alert = QCAlert.objects.prefetch_related("decisions").get(id=alert.id)
        return Response(serialize_alert(alert, include_audit=True))
