from rest_framework import serializers


class VerifiedErrorSerializer(serializers.Serializer):
    error_id = serializers.CharField()
    type = serializers.ChoiceField(
        choices=("POSSIBLE_MISSING_OBJECT", "POSSIBLE_WRONG_CLASS")
    )
    frame = serializers.IntegerField(min_value=0)
    source = serializers.ChoiceField(choices=("synthetic", "natural"), default="synthetic")


class BenchmarkCreateSerializer(serializers.Serializer):
    evaluation_run_id = serializers.IntegerField(min_value=1)
    seed = serializers.IntegerField(min_value=0, default=42)
    top_k = serializers.ListField(
        child=serializers.IntegerField(min_value=1), min_length=1, max_length=10, default=[20, 50]
    )
    verified_errors = VerifiedErrorSerializer(many=True, min_length=1)

    def validate_top_k(self, value):
        return sorted(set(value))
