from rest_framework import serializers
from .models import MRIStudy


class MRIStudySerializer(serializers.ModelSerializer):
    class Meta:
        model = MRIStudy
        fields = '__all__'

