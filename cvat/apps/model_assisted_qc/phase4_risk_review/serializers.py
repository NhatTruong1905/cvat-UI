from rest_framework import serializers


class QueueCreateSerializer(serializers.Serializer):
    evaluation_run_id = serializers.IntegerField(min_value=1)
    job_id = serializers.IntegerField(min_value=1)
    match_iou = serializers.FloatField(min_value=0.01, max_value=1, default=0.5)
    min_confidence = serializers.FloatField(min_value=0, max_value=1, default=0.5)
    dedup_iou = serializers.FloatField(min_value=0.01, max_value=1, default=0.9)
    condition_reliability = serializers.FloatField(min_value=0, max_value=1, default=1.0)


class AlertFilterSerializer(serializers.Serializer):
    status = serializers.ChoiceField(
        choices=("PENDING", "IN_REVIEW", "CONFIRMED", "REJECTED", "RESOLVED"),
        required=False,
    )
    evaluation_run_id = serializers.IntegerField(min_value=1, required=False)
    job_id = serializers.IntegerField(min_value=1, required=False)
