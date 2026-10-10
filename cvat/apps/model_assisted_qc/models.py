from django.conf import settings
from django.db import models

from cvat.apps.engine.models import Job, Task


class DatasetConfiguration(models.Model):
    class ValidationStatus(models.TextChoices):
        NOT_VALIDATED = "not_validated", "Not validated"
        PASSED = "passed", "Passed"
        WARNING = "warning", "Warning"
        BLOCKED = "blocked", "Blocked"

    task = models.OneToOneField(Task, on_delete=models.CASCADE, related_name="evaluation_dataset")
    ground_truth_job = models.ForeignKey(Job, null=True, blank=True, on_delete=models.SET_NULL)
    sampling_method = models.CharField(max_length=32)
    sampled_frames = models.JSONField(default=list)
    conditions = models.JSONField(default=dict)
    validation_status = models.CharField(
        max_length=16,
        choices=ValidationStatus.choices,
        default=ValidationStatus.NOT_VALIDATED,
    )
    validation_result = models.JSONField(default=dict)
    validated_date = models.DateTimeField(null=True, blank=True)
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
    provenance = models.JSONField(default=dict)
    cache_key = models.CharField(max_length=64, blank=True, db_index=True)
    error = models.TextField(blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_date = models.DateTimeField(auto_now_add=True)
    finished_date = models.DateTimeField(null=True, blank=True)


class ThresholdAnalysis(models.Model):
    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name="threshold_analyses")
    ground_truth_job = models.ForeignKey(Job, null=True, on_delete=models.SET_NULL)
    model_path = models.CharField(max_length=1024)
    confidence_values = models.JSONField(default=list)
    iou_values = models.JSONField(default=list)
    objective = models.CharField(max_length=32, default="map50_95")
    conditions = models.JSONField(default=dict)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.RUNNING)
    results = models.JSONField(default=list)
    best_result = models.JSONField(default=dict)
    error = models.TextField(blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_date = models.DateTimeField(auto_now_add=True)
    finished_date = models.DateTimeField(null=True, blank=True)


class QCAlert(models.Model):
    class Type(models.TextChoices):
        MISSING_OBJECT = "POSSIBLE_MISSING_OBJECT", "Possible missing object"
        WRONG_CLASS = "POSSIBLE_WRONG_CLASS", "Possible wrong class"
        EXTRA_OBJECT = "POSSIBLE_EXTRA_OBJECT", "Possible extra or incorrect object"

    class Status(models.TextChoices):
        PENDING = "PENDING", "Pending"
        IN_REVIEW = "IN_REVIEW", "In review"
        CONFIRMED = "CONFIRMED", "Confirmed"
        REJECTED = "REJECTED", "Rejected"
        RESOLVED = "RESOLVED", "Resolved"

    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name="qc_alerts")
    evaluation_run = models.ForeignKey(
        EvaluationRun, on_delete=models.CASCADE, related_name="qc_alerts"
    )
    source_job = models.ForeignKey(
        Job, null=True, on_delete=models.CASCADE, related_name="model_assisted_qc_alerts"
    )
    stable_key = models.CharField(max_length=64)
    frame = models.PositiveIntegerField()
    image_name = models.CharField(max_length=1024, blank=True)
    type = models.CharField(max_length=32, choices=Type.choices)
    prediction = models.JSONField(default=dict)
    annotation = models.JSONField(null=True, blank=True)
    iou = models.FloatField(null=True, blank=True)
    risk_score = models.FloatField()
    score_breakdown = models.JSONField(default=dict)
    reason_codes = models.JSONField(default=list)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.PENDING)
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("evaluation_run", "stable_key"), name="unique_qc_alert_per_run"
            )
        ]
        ordering = ("-risk_score", "frame", "id")


class ReviewDecision(models.Model):
    class Decision(models.TextChoices):
        START_REVIEW = "START_REVIEW", "Start review"
        CONFIRMED = "CONFIRMED", "Confirmed"
        REJECTED = "REJECTED", "Rejected"
        RESOLVED = "RESOLVED", "Resolved"
        NEEDS_MORE_INFO = "NEEDS_MORE_INFO", "Needs more information"

    alert = models.ForeignKey(QCAlert, on_delete=models.CASCADE, related_name="decisions")
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    decision = models.CharField(max_length=32, choices=Decision.choices)
    previous_status = models.CharField(max_length=16, choices=QCAlert.Status.choices)
    resulting_status = models.CharField(max_length=16, choices=QCAlert.Status.choices)
    notes = models.TextField(blank=True)
    created_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("created_date", "id")


class QCBenchmark(models.Model):
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name="qc_benchmarks")
    evaluation_run = models.ForeignKey(
        EvaluationRun, on_delete=models.CASCADE, related_name="qc_benchmarks"
    )
    seed = models.PositiveIntegerField(default=42)
    top_k = models.JSONField(default=list)
    verified_errors = models.JSONField(default=list)
    metrics = models.JSONField(default=dict)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, on_delete=models.SET_NULL)
    created_date = models.DateTimeField(auto_now_add=True)
