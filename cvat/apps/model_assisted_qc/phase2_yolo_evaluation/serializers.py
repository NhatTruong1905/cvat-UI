from rest_framework import serializers


class EvaluationCreateSerializer(serializers.Serializer):
    model_path = serializers.CharField(max_length=1024)
    confidence = serializers.FloatField(min_value=0, max_value=1, default=0.25)
    iou_threshold = serializers.FloatField(min_value=0, max_value=1, default=0.5)
