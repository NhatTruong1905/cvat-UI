from django.utils import timezone
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..common import get_task
from ..models import DatasetConfiguration, ThresholdAnalysis
from ..phase2_yolo_evaluation.services import run_ultralytics
from .engine import run_threshold_analysis, summarize_by_condition
from .serializers import ThresholdAnalysisCreateSerializer


class ThresholdAnalysisView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, task_id):
        task = get_task(request, task_id)
        analyses = ThresholdAnalysis.objects.filter(task=task).order_by("-created_date")[:20]
        return Response(
            [
                {
                    "id": analysis.id,
                    "status": analysis.status,
                    "model_path": analysis.model_path,
                    "objective": analysis.objective,
                    "conditions": analysis.conditions,
                    "results": analysis.results,
                    "best_result": analysis.best_result,
                    "error": analysis.error,
                    "created_date": analysis.created_date,
                }
                for analysis in analyses
            ]
        )

    def post(self, request, task_id):
        task = get_task(request, task_id)
        serializer = ThresholdAnalysisCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        config = DatasetConfiguration.objects.filter(task=task).first()
        if config is None or config.ground_truth_job is None:
            return Response(
                {"error": "Create the Ground Truth dataset before threshold analysis"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if config.validation_status not in {
            DatasetConfiguration.ValidationStatus.PASSED,
            DatasetConfiguration.ValidationStatus.WARNING,
        }:
            return Response(
                {"error": "Ground Truth must pass validation before threshold analysis"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        options = dict(serializer.validated_data)
        analysis = ThresholdAnalysis.objects.create(
            task=task,
            ground_truth_job=config.ground_truth_job,
            conditions=config.conditions,
            created_by=request.user,
            **options,
        )
        try:
            analysis.results, analysis.best_result = run_threshold_analysis(
                ground_truth_job_id=config.ground_truth_job_id,
                conditions=config.conditions,
                evaluation_runner=run_ultralytics,
                **options,
            )
            analysis.status = ThresholdAnalysis.Status.COMPLETED
        except Exception as exc:
            analysis.status = ThresholdAnalysis.Status.FAILED
            analysis.error = str(exc)
        analysis.finished_date = timezone.now()
        analysis.save()
        response_status = (
            status.HTTP_201_CREATED
            if analysis.status == ThresholdAnalysis.Status.COMPLETED
            else status.HTTP_400_BAD_REQUEST
        )
        return Response(
            {
                "id": analysis.id,
                "status": analysis.status,
                "results": analysis.results,
                "best_result": analysis.best_result,
                "error": analysis.error,
            },
            status=response_status,
        )


class ConditionSummaryView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        analyses = ThresholdAnalysis.objects.filter(
            created_by=request.user, status=ThresholdAnalysis.Status.COMPLETED
        ).only("conditions", "best_result")
        return Response(summarize_by_condition(analyses))
