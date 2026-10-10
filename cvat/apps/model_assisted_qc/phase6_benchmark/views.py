from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..common import get_task
from ..models import EvaluationRun, QCBenchmark, QCAlert
from .metrics import benchmark_alerts
from .serializers import BenchmarkCreateSerializer


class BenchmarkView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, task_id):
        task = get_task(request, task_id)
        benchmarks = QCBenchmark.objects.filter(task=task).order_by("-created_date")[:20]
        return Response(
            [
                {
                    "id": benchmark.id,
                    "evaluation_run_id": benchmark.evaluation_run_id,
                    "seed": benchmark.seed,
                    "top_k": benchmark.top_k,
                    "metrics": benchmark.metrics,
                    "created_date": benchmark.created_date,
                }
                for benchmark in benchmarks
            ]
        )

    def post(self, request, task_id):
        task = get_task(request, task_id)
        serializer = BenchmarkCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        run = EvaluationRun.objects.filter(
            id=data["evaluation_run_id"], task=task, status=EvaluationRun.Status.COMPLETED
        ).first()
        if run is None:
            return Response(
                {"error": "A completed evaluation run for this task is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        alerts = list(
            QCAlert.objects.filter(evaluation_run=run).values(
                "id", "type", "frame", "risk_score"
            )
        )
        metrics = benchmark_alerts(
            alerts,
            data["verified_errors"],
            top_k=data["top_k"],
            seed=data["seed"],
        )
        benchmark = QCBenchmark.objects.create(
            task=task,
            evaluation_run=run,
            seed=data["seed"],
            top_k=data["top_k"],
            verified_errors=data["verified_errors"],
            metrics=metrics,
            created_by=request.user,
        )
        return Response({"id": benchmark.id, "metrics": metrics}, status=status.HTTP_201_CREATED)
