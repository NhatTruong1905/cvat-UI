from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("yolo_evaluation", "0005_qc_review_benchmark")]

    operations = [
        migrations.AlterField(
            model_name="qcalert",
            name="type",
            field=models.CharField(
                choices=[
                    ("POSSIBLE_MISSING_OBJECT", "Possible missing object"),
                    ("POSSIBLE_WRONG_CLASS", "Possible wrong class"),
                    ("POSSIBLE_EXTRA_OBJECT", "Possible extra or incorrect object"),
                ],
                max_length=32,
            ),
        ),
    ]
