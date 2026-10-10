from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("engine", "0108_profile_registration_survey_fields"),
        ("yolo_evaluation", "0006_qcalert_extra_object"),
    ]

    operations = [
        migrations.AddField(
            model_name="qcalert",
            name="source_job",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="model_assisted_qc_alerts",
                to="engine.job",
            ),
        ),
    ]
