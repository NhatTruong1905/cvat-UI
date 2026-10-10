from rest_framework import serializers


class ReviewDecisionSerializer(serializers.Serializer):
    decision = serializers.ChoiceField(
        choices=("START_REVIEW", "CONFIRMED", "REJECTED", "RESOLVED", "NEEDS_MORE_INFO")
    )
    notes = serializers.CharField(required=False, allow_blank=True, default="")
