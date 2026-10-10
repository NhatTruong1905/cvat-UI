from django.utils import timezone
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from cvat.apps.dataset_manager.task import get_job_data
from cvat.apps.engine.models import JobType
from cvat.apps.engine.serializers import JobWriteSerializer

from ..common import get_task
from ..models import DatasetConfiguration, EvaluationRun
from .serializers import DatasetCreateSerializer
from .validate_ground_truth import validate_ground_truth


class DatasetView(APIView):
    permission_classes = (IsAuthenticated,)

    def get(self, request, task_id):
        task = get_task(request, task_id)
        config = DatasetConfiguration.objects.filter(task=task).first()
        runs = EvaluationRun.objects.filter(task=task).order_by("-created_date")[:20]
        return Response(
            {
                "dataset": None
                if config is None or config.ground_truth_job_id is None
                else {
                    "ground_truth_job_id": config.ground_truth_job_id,
                    "sampling_method": config.sampling_method,
                    "sampled_frames": config.sampled_frames,
                    "conditions": config.conditions,
                    "validation_status": config.validation_status,
                    "validation_result": config.validation_result,
                    "validated_date": config.validated_date,
                    "updated_date": config.updated_date,
                },
                "runs": [
                    {
                        "id": run.id,
                        "status": run.status,
                        "metrics": run.metrics,
                        "error": run.error,
                        "model_path": run.model_path,
                        "provenance": run.provenance,
                        "created_date": run.created_date,
                    }
                    for run in runs
                ],
            }
        )

    def post(self, request, task_id):
        task = get_task(request, task_id)
        serializer = DatasetCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        payload = {
            "task_id": task.id,
            "type": JobType.GROUND_TRUTH,
            "frame_selection_method": data["sampling_method"],
        }
        if data["sampling_method"] == "manual":
            payload["frames"] = data["frames"]
        else:
            payload.update(frame_count=data["frame_count"], random_seed=data["random_seed"])
        job_serializer = JobWriteSerializer(data=payload, context={"request": request})
        job_serializer.is_valid(raise_exception=True)
        job = job_serializer.save()
        sampled_frames = sorted(job.segment.frame_set)
        config, _ = DatasetConfiguration.objects.update_or_create(
            task=task,
            defaults={
                "ground_truth_job": job,
                "sampling_method": data["sampling_method"],
                "sampled_frames": sampled_frames,
                "conditions": data["conditions"],
                "created_by": request.user,
                "validation_status": DatasetConfiguration.ValidationStatus.NOT_VALIDATED,
                "validation_result": {},
                "validated_date": None,
            },
        )
        return Response(
            {"ground_truth_job_id": job.id, "sampled_frames": config.sampled_frames},
            status=status.HTTP_201_CREATED,
        )


class GroundTruthValidationView(APIView):
    permission_classes = (IsAuthenticated,)

    def post(self, request, task_id):
        task = get_task(request, task_id)
        config = DatasetConfiguration.objects.filter(task=task).first()
        if config is None or config.ground_truth_job is None:
            return Response(
                {"error": "Create the Ground Truth dataset before validating it"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        labels = {label.id: label.name for label in task.get_labels()}
        image_sizes = {
            image.frame: (image.width, image.height)
            for image in task.data.images.filter(frame__in=config.sampled_frames)
        }
        result = validate_ground_truth(
            get_job_data(config.ground_truth_job_id), labels, image_sizes, config.sampled_frames
        )
        config.validation_status = result["status"]
        config.validation_result = result
        config.validated_date = timezone.now()
        config.save(
            update_fields=["validation_status", "validation_result", "validated_date", "updated_date"]
        )
        return Response(result)
