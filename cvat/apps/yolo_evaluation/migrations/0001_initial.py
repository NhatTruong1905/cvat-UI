from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True
    dependencies = [("engine", "0108_profile_registration_survey_fields"), migrations.swappable_dependency(settings.AUTH_USER_MODEL)]
    operations = [
        migrations.CreateModel(name="DatasetConfiguration", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("sampling_method", models.CharField(max_length=32)),
            ("sampled_frames", models.JSONField(default=list)),
            ("conditions", models.JSONField(default=dict)),
            ("updated_date", models.DateTimeField(auto_now=True)),
            ("created_by", models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ("ground_truth_job", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to="engine.job")),
            ("task", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="evaluation_dataset", to="engine.task")),
        ]),
        migrations.CreateModel(name="EvaluationRun", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("model_path", models.CharField(max_length=1024)),
            ("confidence", models.FloatField(default=0.25)),
            ("iou_threshold", models.FloatField(default=0.5)),
            ("status", models.CharField(choices=[("running", "Running"), ("completed", "Completed"), ("failed", "Failed")], default="running", max_length=16)),
            ("metrics", models.JSONField(default=dict)),
            ("predictions", models.JSONField(default=list)),
            ("error", models.TextField(blank=True)),
            ("created_date", models.DateTimeField(auto_now_add=True)),
            ("finished_date", models.DateTimeField(blank=True, null=True)),
            ("created_by", models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL)),
            ("ground_truth_job", models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to="engine.job")),
            ("task", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="yolo_evaluations", to="engine.task")),
        ]),
    ]
