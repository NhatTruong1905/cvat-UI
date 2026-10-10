from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("yolo_evaluation", "0003_thresholdanalysis")]

    operations = [
        migrations.AddField(
            model_name="evaluationrun",
            name="cache_key",
            field=models.CharField(blank=True, db_index=True, max_length=64),
        ),
        migrations.AddField(
            model_name="evaluationrun",
            name="provenance",
            field=models.JSONField(default=dict),
        ),
    ]
