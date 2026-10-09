from django.urls import path

from .views import DatasetView, EvaluationView, GroundTruthValidationView, PredictionView

urlpatterns = [
    path("tasks/<int:task_id>/dataset", DatasetView.as_view()),
    path("tasks/<int:task_id>/validate-ground-truth", GroundTruthValidationView.as_view()),
    path("tasks/<int:task_id>/runs", EvaluationView.as_view()),
    path("runs/<int:run_id>/predictions", PredictionView.as_view()),
]
