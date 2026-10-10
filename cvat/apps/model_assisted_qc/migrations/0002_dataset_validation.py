from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("yolo_evaluation", "0001_initial")]

    operations = [
        migrations.AddField(
            model_name="datasetconfiguration",
            name="validation_status",
            field=models.CharField(
                choices=[
                    ("not_validated", "Not validated"),
                    ("passed", "Passed"),
                    ("warning", "Warning"),
                    ("blocked", "Blocked"),
                ],
                default="not_validated",
                max_length=16,
            ),
        ),
        migrations.AddField(
            model_name="datasetconfiguration",
            name="validation_result",
            field=models.JSONField(default=dict),
        ),
        migrations.AddField(
            model_name="datasetconfiguration",
            name="validated_date",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
