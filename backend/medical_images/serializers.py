from rest_framework import serializers
from .models import MedicalImage

class MedicalImageSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.name', read_only=True)
    image_url = serializers.SerializerMethodField()
    
    class Meta:
        model = MedicalImage
        fields = '__all__'
        read_only_fields = ('created_at',)
    
    def get_image_url(self, obj):
        if obj.image_file:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(obj.image_file.url)
            return obj.image_file.url
        return None
