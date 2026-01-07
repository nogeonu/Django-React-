"""
YOLO 디텍션 API Views
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import requests
import io
from PIL import Image
import pydicom
import numpy as np
import logging

from .orthanc_client import OrthancClient

logger = logging.getLogger(__name__)

# YOLO API 서버 URL
YOLO_API_URL = "http://localhost:5005"


def dicom_to_pil_image(dicom_bytes):
    """DICOM 바이트를 PIL Image로 변환"""
    try:
        dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
        pixel_array = dicom.pixel_array
        
        # 정규화 (0-255)
        pixel_array = pixel_array.astype(float)
        if pixel_array.max() > pixel_array.min():
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255)
        pixel_array = pixel_array.astype(np.uint8)
        
        # PIL Image로 변환
        if len(pixel_array.shape) == 2:
            image = Image.fromarray(pixel_array, mode='L')
        else:
            image = Image.fromarray(pixel_array)
        
        return image
    except Exception as e:
        logger.error(f"DICOM to PIL 변환 오류: {e}")
        raise


@api_view(['POST'])
def yolo_detection(request, instance_id):
    """
    YOLO 디텍션 실행
    
    POST /api/mri/yolo/instances/<instance_id>/detect/
    """
    try:
        logger.info(f"🔍 YOLO 디텍션 시작: instance_id={instance_id}")
        
        # 1. Orthanc에서 DICOM 이미지 가져오기
        client = OrthancClient()
        dicom_data = client.get_instance_file(instance_id)
        
        # 2. DICOM → PIL Image 변환
        pil_image = dicom_to_pil_image(dicom_data)
        
        # 3. PNG 바이트로 변환
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # 4. YOLO API 호출
        logger.info(f"📡 YOLO API 호출: {YOLO_API_URL}/detect")
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        
        yolo_response = requests.post(
            f"{YOLO_API_URL}/detect",
            files=files,
            timeout=60
        )
        
        yolo_response.raise_for_status()
        yolo_result = yolo_response.json()
        
        # 5. 결과 반환
        response_data = {
            'success': yolo_result.get('success', True),
            'instance_id': instance_id,
            'detections': yolo_result.get('detections', []),
            'detection_count': yolo_result.get('count', 0),
            'image_size': yolo_result.get('image_size', []),
        }
        
        logger.info(f"✅ 디텍션 완료: {response_data['detection_count']}개 객체 발견")
        return Response(response_data)
        
    except requests.exceptions.Timeout:
        logger.error("⏱️ YOLO API 타임아웃")
        return Response({
            'success': False,
            'instance_id': instance_id,
            'detections': [],
            'detection_count': 0,
            'error': 'YOLO API 타임아웃 (60초 초과)'
        }, status=status.HTTP_504_GATEWAY_TIMEOUT)
        
    except requests.exceptions.ConnectionError:
        logger.error("🔌 YOLO API 연결 실패")
        return Response({
            'success': False,
            'instance_id': instance_id,
            'detections': [],
            'detection_count': 0,
            'error': 'YOLO API 서버에 연결할 수 없습니다'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
    except Exception as e:
        logger.error(f"❌ 디텍션 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'instance_id': instance_id,
            'detections': [],
            'detection_count': 0,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def yolo_health(request):
    """
    YOLO API 서버 헬스 체크
    
    GET /api/mri/yolo/health/
    """
    try:
        response = requests.get(f"{YOLO_API_URL}/", timeout=5)
        response.raise_for_status()
        yolo_health = response.json()
        
        return Response({
            'success': True,
            'status': 'healthy' if yolo_health.get('model_loaded') else 'unavailable',
            'model_loaded': yolo_health.get('model_loaded', False)
        })
    except Exception as e:
        return Response({
            'success': False,
            'status': 'unavailable',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

