from rest_framework import viewsets, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django_filters.rest_framework import DjangoFilterBackend
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
import os
import requests
import base64
from io import BytesIO
from PIL import Image
from .models import MedicalImage, AIAnalysisResult
from .serializers import MedicalImageSerializer

# 딥러닝 서비스 URL
# 환경 변수 우선, 없으면 기본값 사용
# GCP 서버에서는 같은 서버 내부 통신이므로 127.0.0.1 사용
DL_SERVICE_URL = os.environ.get('DL_SERVICE_URL', 'http://127.0.0.1:5003')

@method_decorator(csrf_exempt, name='dispatch')
class MedicalImageViewSet(viewsets.ModelViewSet):
    queryset = MedicalImage.objects.all()
    serializer_class = MedicalImageSerializer
    authentication_classes = []
    permission_classes = [AllowAny]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['image_type', 'patient_id']
    search_fields = ['description', 'doctor_notes', 'patient_id']
    ordering_fields = ['taken_date', 'created_at']
    ordering = ['-taken_date']
    
    def get_serializer_context(self):
        context = super().get_serializer_context()
        context.update({
            'request': self.request
        })
        return context
    
    @action(detail=True, methods=['post'])
    def analyze(self, request, pk=None):
        """
        의료 이미지 AI 분석 엔드포인트
        POST /api/medical-images/{id}/analyze/
        """
        try:
            medical_image = self.get_object()
            
            if not medical_image.image_file:
                return Response(
                    {'error': '이미지 파일이 없습니다.'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # 이미지 파일 읽기
            image_url = None
            image_base64 = None
            
            # 로컬 환경에서는 항상 파일을 직접 읽기 시도
            # 방법 1: image_file.path 사용
            try:
                image_path = medical_image.image_file.path
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            except (AttributeError, ValueError, OSError):
                # image_file.path가 없는 경우, MEDIA_ROOT에서 직접 찾기
                try:
                    if medical_image.image_file.name:
                        media_path = os.path.join(settings.MEDIA_ROOT, medical_image.image_file.name)
                        if os.path.exists(media_path):
                            with open(media_path, 'rb') as f:
                                image_bytes = f.read()
                                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                except Exception:
                    pass
            
            # 로컬에서 파일을 찾지 못한 경우, 로컬 Django 서버 URL 사용
            if not image_base64:
                # 로컬 개발 환경인지 확인
                if settings.DEBUG:
                    # 로컬 Django 서버 URL 사용
                    try:
                        image_url = request.build_absolute_uri(medical_image.image_file.url)
                    except:
                        image_url = f"http://localhost:8000{medical_image.image_file.url}"
                else:
                    # 프로덕션 환경
                    image_url = f"{settings.PRODUCTION_DOMAIN}{medical_image.image_file.url}"
            
            # base64 또는 URL이 없으면 에러
            if not image_base64 and not image_url:
                return Response(
                    {
                        'error': '이미지 파일을 읽을 수 없습니다.',
                        'detail': f'파일명: {getattr(medical_image.image_file, "name", "N/A")}, MEDIA_ROOT: {settings.MEDIA_ROOT}'
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # 딥러닝 서비스 호출
            try:
                # mosec 서비스는 /inference 엔드포인트 사용
                if image_url:
                    payload = {
                        'image_url': image_url,
                        'patient_id': medical_image.patient_id,
                        'metadata': {
                            'image_type': medical_image.image_type,
                            'image_id': str(medical_image.id)
                        }
                    }
                else:
                    payload = {
                        'image_data': image_base64,
                        'patient_id': medical_image.patient_id,
                        'metadata': {
                            'image_type': medical_image.image_type,
                            'image_id': str(medical_image.id)
                        }
                    }
                
                response = requests.post(
                    f'{DL_SERVICE_URL}/inference',
                    json=payload,
                    timeout=60  # 딥러닝 추론은 시간이 걸릴 수 있음
                )
                
                if response.status_code != 200:
                    return Response(
                        {
                            'error': f'딥러닝 서비스 오류: {response.status_code}',
                            'detail': response.text
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                result = response.json()
                
                if not result.get('success'):
                    return Response(
                        {'error': result.get('error', '분석 실패')},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR
                    )
                
                analysis_data = result.get('data', {})
                
                # 분석 결과 저장
                analysis_result = AIAnalysisResult.objects.create(
                    image=medical_image,
                    analysis_type='BREAST_MRI',  # 모델에 맞게 조정
                    results=analysis_data.get('probabilities', {}),
                    confidence=analysis_data.get('confidence'),
                    findings=analysis_data.get('findings', ''),
                    recommendations=analysis_data.get('recommendations', ''),
                    model_version=analysis_data.get('model_version', '1.0.0')
                )
                
                # 시리얼라이저로 응답 반환
                serializer = self.get_serializer(medical_image)
                return Response(serializer.data, status=status.HTTP_200_OK)
                
            except requests.exceptions.ConnectionError as e:
                # 로컬/프로덕션 환경에 따른 해결 방법 안내
                if settings.DEBUG:
                    solution = '로컬 개발 환경: 다음 명령어로 mosec 서비스를 실행하세요:\ncd backend/dl_service && python3 app.py'
                else:
                    solution = '프로덕션 환경: GCP 서버에서 mosec 서비스 상태를 확인하세요:\nsudo systemctl status dl-service\nsudo systemctl restart dl-service'
                
                return Response(
                    {
                        'error': '딥러닝 서비스에 연결할 수 없습니다.',
                        'detail': f'mosec 서비스가 실행되지 않았습니다. (URL: {DL_SERVICE_URL})',
                        'solution': solution
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            except requests.exceptions.Timeout as e:
                return Response(
                    {
                        'error': '딥러닝 서비스 응답 시간 초과',
                        'detail': '모델 추론에 시간이 너무 오래 걸립니다.'
                    },
                    status=status.HTTP_504_GATEWAY_TIMEOUT
                )
            except requests.exceptions.RequestException as e:
                return Response(
                    {
                        'error': f'딥러닝 서비스 연결 실패: {str(e)}',
                        'detail': 'mosec 서비스가 실행 중인지 확인하세요.'
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            except Exception as e:
                return Response(
                    {'error': f'분석 중 오류 발생: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
        except MedicalImage.DoesNotExist:
            return Response(
                {'error': '의료 이미지를 찾을 수 없습니다.'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': f'예상치 못한 오류: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
