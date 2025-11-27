from rest_framework import serializers
from django.conf import settings
from .models import MedicalImage, AIAnalysisResult

class AIAnalysisResultSerializer(serializers.ModelSerializer):
    """AI 분석 결과 시리얼라이저"""
    class Meta:
        model = AIAnalysisResult
        fields = '__all__'
        read_only_fields = ('analysis_date',)

class MedicalImageSerializer(serializers.ModelSerializer):
    patient_name = serializers.SerializerMethodField()
    image_url = serializers.SerializerMethodField()
    analysis_results = AIAnalysisResultSerializer(many=True, read_only=True)
    
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
        """
        이미지 URL 생성 - API 엔드포인트 사용 (한국어 파일명 문제 해결)
        """
        if obj.image_file:
            request = self.context.get('request')
            if request:
                # API 엔드포인트를 통해 이미지 서빙 (한국어 파일명 문제 해결)
                url = request.build_absolute_uri(f'/api/medical-images/{obj.id}/image/')
                # localhost를 프로덕션 도메인으로 변경
                if 'localhost' in url or '127.0.0.1' in url:
                    url = url.replace('http://localhost:8000', settings.PRODUCTION_DOMAIN)
                    url = url.replace('http://127.0.0.1:8000', settings.PRODUCTION_DOMAIN)
                return url
            # request가 없을 경우 프로덕션 도메인 사용
            return f"{settings.PRODUCTION_DOMAIN}/api/medical-images/{obj.id}/image/"
        return None
