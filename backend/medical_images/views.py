from rest_framework import viewsets, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django_filters.rest_framework import DjangoFilterBackend
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from django.http import FileResponse, Http404
import os
import requests
import base64
import traceback
import logging
from urllib.parse import unquote
from .models import MedicalImage, AIAnalysisResult
from .serializers import MedicalImageSerializer

logger = logging.getLogger(__name__)

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
    
    @action(detail=True, methods=['get'])
    def image(self, request, pk=None):
        """
        이미지 파일 직접 서빙 엔드포인트
        GET /api/medical-images/{id}/image/
        한국어 파일명 문제를 해결하기 위해 이미지를 직접 서빙
        """
        try:
            medical_image = self.get_object()
            logger.info(f"이미지 서빙 요청: ID={pk}, 파일명={medical_image.image_file.name if medical_image.image_file else 'None'}")
            
            if not medical_image.image_file:
                logger.warning(f"이미지 파일이 없음: ID={pk}")
                raise Http404("이미지 파일이 없습니다.")
            
            # 이미지 파일 경로 찾기
            image_path = None
            
            # 방법 1: image_file.path 사용
            try:
                if hasattr(medical_image.image_file, 'path'):
                    image_path = medical_image.image_file.path
                    if os.path.exists(image_path):
                        logger.info(f"이미지 파일 찾음 (path): {image_path}")
                        ext = os.path.splitext(image_path)[1].lower()
                        content_type_map = {
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.png': 'image/png',
                            '.gif': 'image/gif',
                            '.bmp': 'image/bmp',
                        }
                        content_type = content_type_map.get(ext, 'image/jpeg')
                        return FileResponse(open(image_path, 'rb'), content_type=content_type)
            except (AttributeError, ValueError, OSError) as e:
                logger.warning(f"image_file.path 접근 실패: {e}")
                pass
            
            # 방법 2: MEDIA_ROOT에서 찾기
            if not image_path:
                file_name = medical_image.image_file.name
                if file_name.startswith('medical_images/'):
                    file_name = file_name.replace('medical_images/', '', 1)
                
                # 여러 경로 시도
                possible_paths = [
                    os.path.join(settings.MEDIA_ROOT, medical_image.image_file.name),
                    os.path.join(settings.MEDIA_ROOT, 'medical_images', file_name),
                    os.path.join(settings.MEDIA_ROOT, 'medical_images', os.path.basename(file_name)),
                    os.path.join(settings.MEDIA_ROOT, 'medical_images', unquote(file_name)),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path) and os.path.isfile(path):
                        image_path = path
                        break
                
                # 파일명 일부로 검색 (최후의 수단)
                if not image_path:
                    medical_images_dir = os.path.join(settings.MEDIA_ROOT, 'medical_images')
                    if os.path.exists(medical_images_dir):
                        base_name = os.path.splitext(os.path.basename(file_name))[0]
                        decoded_base_name = os.path.splitext(os.path.basename(unquote(file_name)))[0]
                        
                        for root, dirs, files in os.walk(medical_images_dir):
                            for f in files:
                                file_base = os.path.splitext(f)[0]
                                if base_name in file_base or decoded_base_name in file_base or file_base in base_name or file_base in decoded_base_name:
                                    full_path = os.path.join(root, f)
                                    if os.path.exists(full_path):
                                        image_path = full_path
                                        break
                            if image_path:
                                break
                
                if image_path and os.path.exists(image_path):
                    # 파일 확장자에 따라 content_type 결정
                    ext = os.path.splitext(image_path)[1].lower()
                    content_type_map = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.bmp': 'image/bmp',
                    }
                    content_type = content_type_map.get(ext, 'image/jpeg')
                    logger.info(f"이미지 파일 찾음 (MEDIA_ROOT): {image_path}, content_type={content_type}")
                    return FileResponse(open(image_path, 'rb'), content_type=content_type)
            
            logger.error(f"이미지 파일을 찾을 수 없음: ID={pk}, 파일명={medical_image.image_file.name}, MEDIA_ROOT={settings.MEDIA_ROOT}")
            raise Http404(f"이미지 파일을 찾을 수 없습니다. (파일명: {medical_image.image_file.name})")
            
        except MedicalImage.DoesNotExist:
            raise Http404("의료 이미지를 찾을 수 없습니다.")
        except Exception as e:
            logger.error(f"이미지 서빙 중 오류: {str(e)}")
            raise Http404(f"이미지를 불러올 수 없습니다: {str(e)}")
    
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
            
            # 이미지 파일 읽기 - base64 우선, URL은 최후의 수단
            image_url = None
            image_base64 = None
            
            # 방법 1: image_file.path 사용 (가장 확실한 방법)
            try:
                image_path = medical_image.image_file.path
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                        logger.info(f"이미지 파일 로드 성공 (path): {image_path}")
            except (AttributeError, ValueError, OSError) as e:
                logger.warning(f"image_file.path 접근 실패: {e}")
                # 방법 2: MEDIA_ROOT에서 직접 찾기
                try:
                    if medical_image.image_file.name:
                        # 파일명 가져오기
                        file_name = medical_image.image_file.name
                        
                        # medical_images/ 제거
                        if file_name.startswith('medical_images/'):
                            file_name = file_name.replace('medical_images/', '', 1)
                        
                        # URL 인코딩된 파일명 디코딩 시도
                        decoded_file_name = unquote(file_name)
                        
                        # 여러 경로 시도 (한국어 파일명, URL 인코딩된 파일명 모두 시도)
                        possible_paths = [
                            os.path.join(settings.MEDIA_ROOT, medical_image.image_file.name),  # 전체 경로
                            os.path.join(settings.MEDIA_ROOT, 'medical_images', file_name),  # 원본 파일명
                            os.path.join(settings.MEDIA_ROOT, 'medical_images', decoded_file_name),  # 디코딩된 파일명
                            os.path.join(settings.MEDIA_ROOT, 'medical_images', os.path.basename(file_name)),  # basename
                            os.path.join(settings.MEDIA_ROOT, 'medical_images', os.path.basename(decoded_file_name)),  # 디코딩된 basename
                        ]
                        
                        # 날짜별 폴더 구조도 시도 (YYYY/MM/DD/파일명)
                        if '/' in file_name:
                            date_parts = file_name.split('/')
                            if len(date_parts) >= 2:
                                date_path = '/'.join(date_parts[:-1])
                                filename_only = date_parts[-1]
                                possible_paths.extend([
                                    os.path.join(settings.MEDIA_ROOT, 'medical_images', date_path, filename_only),
                                    os.path.join(settings.MEDIA_ROOT, 'medical_images', date_path, unquote(filename_only)),
                                ])
                        
                        # MEDIA_ROOT의 medical_images 디렉토리에서 모든 파일 검색 (최후의 수단)
                        medical_images_dir = os.path.join(settings.MEDIA_ROOT, 'medical_images')
                        if os.path.exists(medical_images_dir):
                            # 파일명의 일부만으로 검색 (확장자 제외)
                            base_name = os.path.splitext(os.path.basename(file_name))[0]
                            decoded_base_name = os.path.splitext(os.path.basename(decoded_file_name))[0]
                            
                            for root, dirs, files in os.walk(medical_images_dir):
                                for f in files:
                                    file_base = os.path.splitext(f)[0]
                                    # 파일명의 일부가 일치하면 시도
                                    if base_name in file_base or decoded_base_name in file_base or file_base in base_name or file_base in decoded_base_name:
                                        full_path = os.path.join(root, f)
                                        if full_path not in possible_paths:
                                            possible_paths.append(full_path)
                        
                        # 모든 경로 시도
                        for media_path in possible_paths:
                            try:
                                if os.path.exists(media_path) and os.path.isfile(media_path):
                                    with open(media_path, 'rb') as f:
                                        image_bytes = f.read()
                                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                        logger.info(f"이미지 파일 로드 성공 (MEDIA_ROOT): {media_path}")
                                        break
                            except Exception as path_error:
                                continue
                                
                except Exception as e2:
                    logger.error(f"MEDIA_ROOT에서 이미지 찾기 실패: {e2}", exc_info=True)
            
            # base64로 로드 실패한 경우에만 URL 사용 (최후의 수단)
            if not image_base64:
                logger.warning(f"파일 시스템에서 이미지를 찾지 못함. URL 사용 시도: {medical_image.image_file.name}")
                # 프로덕션 환경에서는 항상 PRODUCTION_DOMAIN 사용
                try:
                    image_url = f"{settings.PRODUCTION_DOMAIN}{medical_image.image_file.url}"
                    logger.info(f"이미지 URL 생성: {image_url}")
                except Exception as e:
                    logger.error(f"이미지 URL 생성 실패: {e}")
            
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
                    solution = '로컬 개발 환경: 다음 명령어로 mosec 서비스를 실행하세요:\ncd backend/breast_ai_service && python3 app.py'
                else:
                    solution = '프로덕션 환경: GCP 서버에서 mosec 서비스 상태를 확인하세요:\nsudo systemctl status breast-ai-service\nsudo systemctl restart breast-ai-service'
                
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
            # 모든 예외를 로깅하고 상세한 에러 메시지 반환
            logger.error(f"AI 분석 중 예기치 않은 오류 발생: {str(e)}", exc_info=True)
            return Response(
                {'error': f'예상치 못한 오류: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
