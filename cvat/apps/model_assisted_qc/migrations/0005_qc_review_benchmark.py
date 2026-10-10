from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("yolo_evaluation", "0004_evaluationrun_cache_provenance"),
    ]

    operations = [
        migrations.CreateModel(
            name="QCAlert",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("stable_key", models.CharField(max_length=64)),
                ("frame", models.PositiveIntegerField()),
                ("image_name", models.CharField(blank=True, max_length=1024)),
                ("type", models.CharField(choices=[("POSSIBLE_MISSING_OBJECT", "Possible missing object"), ("POSSIBLE_WRONG_CLASS", "Possible wrong class")], max_length=32)),
                ("prediction", models.JSONField(default=dict)),
                ("annotation", models.JSONField(blank=True, null=True)),
                ("iou", models.FloatField(blank=True, null=True)),
                ("risk_score", models.FloatField()),
                ("score_breakdown", models.JSONField(default=dict)),
                ("reason_codes", models.JSONField(default=list)),
                ("status", models.CharField(choices=[("PENDING", "Pending"), ("IN_REVIEW", "In review"), ("CONFIRMED", "Confirmed"), ("REJECTED", "Rejected"), ("RESOLVED", "Resolved")], default="PENDING", max_length=16)),
                ("created_date", models.DateTimeField(auto_now_add=True)),
                ("updated_date", models.DateTimeField(auto_now=True)),
                ("evaluation_run", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="qc_alerts", to="yolo_evaluation.evaluationrun")),
                ("task", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="qc_alerts", to="engine.task")),
            ],
            options={"ordering": ("-risk_score", "frame", "id")},
        ),
        migrations.CreateModel(
            name="ReviewDecision",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("decision", models.CharField(choices=[("START_REVIEW", "Start review"), ("CONFIRMED", "Confirmed"), ("REJECTED", "Rejected"), ("RESOLVED", "Resolved"), ("NEEDS_MORE_INFO", "Needs more information")], max_length=32)),
                ("previous_status", models.CharField(choices=[("PENDING", "Pending"), ("IN_REVIEW", "In review"), ("CONFIRMED", "Confirmed"), ("REJECTED", "Rejected"), ("RESOLVED", "Resolved")], max_length=16)),
                ("resulting_status", models.CharField(choices=[("PENDING", "Pending"), ("IN_REVIEW", "In review"), ("CONFIRMED", "Confirmed"), ("REJECTED", "Rejected"), ("RESOLVED", "Resolved")], max_length=16)),
                ("notes", models.TextField(blank=True)),
                ("created_date", models.DateTimeField(auto_now_add=True)),
                ("alert", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="decisions", to="yolo_evaluation.qcalert")),
                ("reviewer", models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ],
            options={"ordering": ("created_date", "id")},
        ),
        migrations.CreateModel(
            name="QCBenchmark",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("seed", models.PositiveIntegerField(default=42)),
                ("top_k", models.JSONField(default=list)),
                ("verified_errors", models.JSONField(default=list)),
                ("metrics", models.JSONField(default=dict)),
                ("created_date", models.DateTimeField(auto_now_add=True)),
                ("created_by", models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
                ("evaluation_run", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="qc_benchmarks", to="yolo_evaluation.evaluationrun")),
                ("task", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="qc_benchmarks", to="engine.task")),
            ],
        ),
        migrations.AddConstraint(
            model_name="qcalert",
            constraint=models.UniqueConstraint(fields=("evaluation_run", "stable_key"), name="unique_qc_alert_per_run"),
        ),
    ]
