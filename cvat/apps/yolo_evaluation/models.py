from django.conf import settings
from django.db import models

from cvat.apps.engine.models import Job, Task


class DatasetConfiguration(models.Model):
    task = models.OneToOneField(Task, on_delete=models.CASCADE, related_name="evaluation_dataset")
    ground_truth_job = models.ForeignKey(Job, null=True, blank=True, on_delete=models.SET_NULL)
    sampling_method = models.CharField(max_length=32)
    sampled_frames = models.JSONField(default=list)
    conditions = models.JSONField(default=dict)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    updated_date = models.DateTimeField(auto_now=True)


class EvaluationRun(models.Model):
    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name="yolo_evaluations")
    ground_truth_job = models.ForeignKey(Job, null=True, on_delete=models.SET_NULL)
    model_path = models.CharField(max_length=1024)
    confidence = models.FloatField(default=0.25)
    iou_threshold = models.FloatField(default=0.5)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.RUNNING)
    metrics = models.JSONField(default=dict)
    predictions = models.JSONField(default=list)
    error = models.TextField(blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_date = models.DateTimeField(auto_now_add=True)
    finished_date = models.DateTimeField(null=True, blank=True)

