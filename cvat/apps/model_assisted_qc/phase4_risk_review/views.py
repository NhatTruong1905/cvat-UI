from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from cvat.apps.engine.models import Job

from ..common import get_task
from ..models import EvaluationRun, QCAlert
from .serializers import AlertFilterSerializer, QueueCreateSerializer
from .services import generate_alert_queue


def serialize_alert(alert, include_audit=False):
    result = {
        "id": alert.id,
        "evaluation_run_id": alert.evaluation_run_id,
        "frame": alert.frame,
        "image_name": alert.image_name,
        "type": alert.type,
        "prediction": alert.prediction,
        "annotation": alert.annotation,
        "iou": alert.iou,
        "risk_score": alert.risk_score,
        "score_breakdown": alert.score_breakdown,
        "reason_codes": alert.reason_codes,
        "status": alert.status,
        "created_date": alert.created_date,
    }
    if include_audit:
        result["audit"] = [
            {
                "decision": decision.decision,
                "previous_status": decision.previous_status,
                "resulting_status": decision.resulting_status,
                "notes": decision.notes,
                "reviewer_id": decision.reviewer_id,
                "created_date": decision.created_date,
            }
            for decision in alert.decisions.all()
        ]
    return result


class AlertQueueView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, task_id):
        task = get_task(request, task_id)
        serializer = AlertFilterSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        alerts = QCAlert.objects.filter(task=task)
        if value := serializer.validated_data.get("status"):
            alerts = alerts.filter(status=value)
        if value := serializer.validated_data.get("evaluation_run_id"):
            alerts = alerts.filter(evaluation_run_id=value)
        if value := serializer.validated_data.get("job_id"):
            alerts = alerts.filter(source_job_id=value)
        return Response([serialize_alert(alert) for alert in alerts[:5000]])

    def post(self, request, task_id):
        task = get_task(request, task_id)
        serializer = QueueCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        options = dict(serializer.validated_data)
        run_id = options.pop("evaluation_run_id")
        job_id = options.pop("job_id")
        run = EvaluationRun.objects.filter(
            id=run_id, task=task, status=EvaluationRun.Status.COMPLETED
        ).first()
        if run is None:
            return Response(
                {"error": "A completed evaluation run for this task is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        source_job = Job.objects.filter(id=job_id, segment__task=task).first()
        if source_job is None:
            return Response(
                {"error": "The annotation job does not belong to this task"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        result = generate_alert_queue(run, source_job, options)
        return Response(result, status=status.HTTP_201_CREATED)


class AlertDetailView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, alert_id):
        alert = QCAlert.objects.prefetch_related("decisions").filter(id=alert_id).first()
        if alert is None:
            return Response({"error": "Alert was not found"}, status=status.HTTP_404_NOT_FOUND)
        get_task(request, alert.task_id)
        return Response(serialize_alert(alert, include_audit=True))
