from django.apps import AppConfig


class ModelAssistedQCConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "cvat.apps.model_assisted_qc"
    label = "yolo_evaluation"
    verbose_name = "Model-assisted QC"
