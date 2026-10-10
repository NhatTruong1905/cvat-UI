from django.utils import timezone
from rest_framework import status
from rest_framework.exceptions import NotFound
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..common import get_task
from ..models import DatasetConfiguration, EvaluationRun
from .serializers import EvaluationCreateSerializer
from .services import build_inference_provenance, run_ultralytics


class EvaluationView(APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, task_id):
        task = get_task(request, task_id)
        serializer = EvaluationCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        config = DatasetConfiguration.objects.filter(task=task).first()
        if config is None or config.ground_truth_job is None:
            return Response(
                {"error": "Create the Ground Truth dataset before running evaluation"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if config.validation_status == DatasetConfiguration.ValidationStatus.NOT_VALIDATED:
            return Response(
                {"error": "Validate the Ground Truth annotations before running evaluation"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if config.validation_status == DatasetConfiguration.ValidationStatus.BLOCKED:
            return Response(
                {"error": "Ground Truth validation has blocker errors. Fix and validate again"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        evaluation_options = dict(serializer.validated_data)
        try:
            provenance, cache_key = build_inference_provenance(
                ground_truth_job_id=config.ground_truth_job_id, **evaluation_options
            )
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        cached_run = EvaluationRun.objects.filter(
            task=task,
            ground_truth_job=config.ground_truth_job,
            cache_key=cache_key,
            status=EvaluationRun.Status.COMPLETED,
        ).first()
        if cached_run:
            return Response(
                {
                    "id": cached_run.id,
                    "status": cached_run.status,
                    "metrics": cached_run.metrics,
                    "error": "",
                    "cached": True,
                    "provenance": cached_run.provenance,
                }
            )
        EvaluationRun.objects.filter(
            task=task,
            status=EvaluationRun.Status.RUNNING,
        ).update(
            status=EvaluationRun.Status.FAILED,
            error="Evaluation was interrupted because the backend worker stopped",
            finished_date=timezone.now(),
        )
        run = EvaluationRun.objects.create(
            task=task,
            ground_truth_job=config.ground_truth_job,
            created_by=request.user,
            provenance=provenance,
            cache_key=cache_key,
            **evaluation_options,
        )
        try:
            run.metrics, run.predictions = run_ultralytics(
                ground_truth_job_id=config.ground_truth_job_id, **evaluation_options
            )
            run.status = EvaluationRun.Status.COMPLETED
        except Exception as exc:
            run.status = EvaluationRun.Status.FAILED
            run.error = str(exc)
        run.finished_date = timezone.now()
        run.save()
        response_status = (
            status.HTTP_201_CREATED
            if run.status == EvaluationRun.Status.COMPLETED
            else status.HTTP_400_BAD_REQUEST
        )
        return Response(
            {
                "id": run.id,
                "status": run.status,
                "metrics": run.metrics,
                "error": run.error,
                "cached": False,
                "provenance": run.provenance,
            },
            status=response_status,
        )


class PredictionView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, run_id):
        try:
            run = EvaluationRun.objects.get(pk=run_id)
        except EvaluationRun.DoesNotExist as exc:
            raise NotFound("Evaluation run was not found") from exc
        get_task(request, run.task_id)
        return Response({"run_id": run.id, "predictions": run.predictions})
