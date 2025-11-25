from rest_framework import serializers
from django.conf import settings
from .models import MedicalImage

class MedicalImageSerializer(serializers.ModelSerializer):
    patient_name = serializers.SerializerMethodField()
    image_url = serializers.SerializerMethodField()
    
    class Meta:
        model = MedicalImage
        fields = '__all__'
        read_only_fields = ('created_at',)
    
    def get_patient_name(self, obj):
        """환자 ID로 환자 이름 조회"""
        try:
            from lung_cancer.models import Patient
            patient = Patient.objects.get(id=obj.patient_id)
            return patient.name
        except:
            return ''
    
    def get_image_url(self, obj):
        if obj.image_file:
            request = self.context.get('request')
            if request:
                url = request.build_absolute_uri(obj.image_file.url)
                # localhost를 프로덕션 도메인으로 변경
                if 'localhost' in url or '127.0.0.1' in url:
                    url = url.replace('http://localhost:8000', settings.PRODUCTION_DOMAIN)
                    url = url.replace('http://127.0.0.1:8000', settings.PRODUCTION_DOMAIN)
                return url
            # request가 없을 경우 프로덕션 도메인 사용
            return f"{settings.PRODUCTION_DOMAIN}{obj.image_file.url}"
        return None
