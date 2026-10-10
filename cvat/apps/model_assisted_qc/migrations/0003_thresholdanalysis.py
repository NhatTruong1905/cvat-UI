from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("engine", "0108_profile_registration_survey_fields"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("yolo_evaluation", "0002_dataset_validation"),
    ]

    operations = [
        migrations.CreateModel(
            name="ThresholdAnalysis",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("model_path", models.CharField(max_length=1024)),
                ("confidence_values", models.JSONField(default=list)),
                ("iou_values", models.JSONField(default=list)),
                ("objective", models.CharField(default="map50_95", max_length=32)),
                ("conditions", models.JSONField(default=dict)),
                ("status", models.CharField(choices=[("running", "Running"), ("completed", "Completed"), ("failed", "Failed")], default="running", max_length=16)),
                ("results", models.JSONField(default=list)),
                ("best_result", models.JSONField(default=dict)),
                ("error", models.TextField(blank=True)),
                ("created_date", models.DateTimeField(auto_now_add=True)),
                ("finished_date", models.DateTimeField(blank=True, null=True)),
                ("created_by", models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
                ("ground_truth_job", models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to="engine.job")),
                ("task", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="threshold_analyses", to="engine.task")),
            ],
        ),
    ]
