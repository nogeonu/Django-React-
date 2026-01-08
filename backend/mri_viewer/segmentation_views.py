"""
MRI 세그멘테이션 API Views
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import requests
import io
import logging
import base64
from .orthanc_client import OrthancClient

logger = logging.getLogger(__name__)

# 세그멘테이션 API 서버 URL
SEGMENTATION_API_URL = "http://localhost:5006"


@api_view(['POST'])
def mri_segmentation(request, instance_id):
    """
    MRI 세그멘테이션 실행 및 Orthanc에 저장
    
    POST /api/mri/segmentation/instances/<instance_id>/segment/
    """
    try:
        logger.info(f"🔍 MRI 세그멘테이션 시작: instance_id={instance_id}")
        
        # 1. Orthanc에서 DICOM 이미지 가져오기
        client = OrthancClient()
        dicom_data = client.get_instance_file(instance_id)
        
        # 2. 세그멘테이션 API 호출
        logger.info(f"📡 세그멘테이션 API 호출: {SEGMENTATION_API_URL}/segment")
        files = {'file': ('image.dcm', io.BytesIO(dicom_data), 'application/dicom')}
        
        seg_response = requests.post(
            f"{SEGMENTATION_API_URL}/segment",
            files=files,
            timeout=120  # 세그멘테이션은 시간이 걸릴 수 있음
        )
        
        seg_response.raise_for_status()
        seg_result = seg_response.json()
        
        if not seg_result.get('success'):
            raise Exception(seg_result.get('error', '세그멘테이션 실패'))
        
        # 3. 결과 반환 (마스크는 base64로 인코딩되어 있음)
        response_data = {
            'success': True,
            'instance_id': instance_id,
            'mask_base64': seg_result.get('mask_base64', ''),
            'tumor_pixels': seg_result.get('tumor_pixels', 0),
            'total_pixels': seg_result.get('total_pixels', 0),
            'tumor_ratio': seg_result.get('tumor_ratio', 0.0),
            'image_size': seg_result.get('image_size', []),
        }
        
        logger.info(f"✅ 세그멘테이션 완료: 종양 비율 {response_data['tumor_ratio']:.2%}")
        return Response(response_data)
        
    except requests.exceptions.Timeout:
        logger.error("⏱️ 세그멘테이션 API 타임아웃")
        return Response({
            'success': False,
            'instance_id': instance_id,
            'error': '세그멘테이션 API 타임아웃 (120초 초과)'
        }, status=status.HTTP_504_GATEWAY_TIMEOUT)
        
    except requests.exceptions.ConnectionError:
        logger.error("🔌 세그멘테이션 API 연결 실패")
        return Response({
            'success': False,
            'instance_id': instance_id,
            'error': '세그멘테이션 API 서버에 연결할 수 없습니다'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
    except Exception as e:
        logger.error(f"❌ 세그멘테이션 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'instance_id': instance_id,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def segmentation_health(request):
    """
    세그멘테이션 API 서버 상태 확인
    
    GET /api/mri/segmentation/health/
    """
    try:
        response = requests.get(f"{SEGMENTATION_API_URL}/", timeout=5)
        response.raise_for_status()
        health = response.json()
        
        return Response({
            'success': True,
            'status': 'healthy',
            'model_loaded': health.get('model_loaded', False),
            'model_type': health.get('model_type', 'Unknown')
        })
    except Exception as e:
        return Response({
            'success': False,
            'status': 'unavailable',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

