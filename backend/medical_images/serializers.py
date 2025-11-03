from rest_framework import serializers
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
            from django.conf import settings
            # GCS를 사용하면 url이 이미 절대 URL임
            if getattr(settings, 'USE_GCS', False):
                url = obj.image_file.url
                print(f"[MedicalImageSerializer] GCS URL: {url}")
                return url
            
            # 로컬 파일 시스템 사용 시
            request = self.context.get('request')
            if request:
                # 프록시/리버스프록시 환경에서 절대 URL 생성
                try:
                    url = request.build_absolute_uri(obj.image_file.url)
                except Exception:
                    url = None
                if not url or url.startswith('http://127.0.0.1') or url.startswith('http://localhost'):
                    base = getattr(settings, 'PUBLIC_BASE_URL', None)
                    if not base:
                        scheme = 'https' if request.is_secure() else 'http'
                        host = request.get_host()
                        base = f"{scheme}://{host}"
                    path = obj.image_file.url
                    if path.startswith('/'):
                        url = base + path
                print(f"[MedicalImageSerializer] Generated local image URL: {url}")
                return url
            # request 컨텍스트가 없을 때는 환경 변수 기반으로 절대 URL 생성
            base = getattr(settings, 'PUBLIC_BASE_URL', None)
            url = obj.image_file.url
            if base and url.startswith('/'):
                url = base + url
            print(f"[MedicalImageSerializer] No request context, using relative URL: {url}")
            return url
        print(f"[MedicalImageSerializer] No image_file for obj: {obj}")
        return None
