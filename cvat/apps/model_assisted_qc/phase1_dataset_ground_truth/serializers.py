from rest_framework import serializers


class DatasetCreateSerializer(serializers.Serializer):
    sampling_method = serializers.ChoiceField(choices=("random_uniform", "manual"))
    frame_count = serializers.IntegerField(min_value=1, required=False)
    frames = serializers.ListField(child=serializers.IntegerField(min_value=0), required=False)
    random_seed = serializers.IntegerField(min_value=0, required=False, default=0)
    conditions = serializers.DictField(required=False, default=dict)

    def validate(self, attrs):
        field = "frames" if attrs["sampling_method"] == "manual" else "frame_count"
        if not attrs.get(field):
            raise serializers.ValidationError({field: "This field is required for the selected method."})
        return attrs
