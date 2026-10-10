from rest_framework import serializers


class ThresholdAnalysisCreateSerializer(serializers.Serializer):
    model_path = serializers.CharField(max_length=1024)
    confidence_values = serializers.ListField(
        child=serializers.FloatField(min_value=0.01, max_value=1), min_length=1, max_length=5
    )
    iou_values = serializers.ListField(
        child=serializers.FloatField(min_value=0.01, max_value=1), min_length=1, max_length=5
    )
    objective = serializers.ChoiceField(
        choices=("map50_95", "map50", "precision", "recall", "f1"), default="map50_95"
    )

    def validate(self, attrs):
        attrs["confidence_values"] = sorted(set(attrs["confidence_values"]))
        attrs["iou_values"] = sorted(set(attrs["iou_values"]))
        if len(attrs["confidence_values"]) * len(attrs["iou_values"]) > 25:
            raise serializers.ValidationError("A threshold analysis is limited to 25 combinations.")
        return attrs
