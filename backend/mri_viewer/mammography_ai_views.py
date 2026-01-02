"""
유방촬영술 AI 디텍션 API Views
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import requests
import base64
import os
import logging
from io import BytesIO
from PIL import Image
import numpy as np
import pydicom

from .orthanc_client import OrthancClient

logger = logging.getLogger(__name__)

# AI 서비스 URL
MAMMOGRAPHY_AI_SERVICE_URL = os.getenv('MAMMOGRAPHY_AI_SERVICE_URL', 'http://localhost:5004')


@api_view(['POST'])
def mammography_ai_detection(request, instance_id):
    """
    유방촬영술 이미지에 대해 YOLO11 디텍션 실행
    
    POST /api/mri/mammography/instances/<instance_id>/detect/
    
    Request body:
    {
        "confidence": 0.25,  # optional
        "iou_threshold": 0.45  # optional
    }
    
    Response:
    {
        "success": true,
        "instance_id": "abc123",
        "detections": [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.85,
                "class_id": 0,
                "class_name": "mass"
            }
        ],
        "annotated_image_base64": "base64_string"
    }
    """
    try:
        # Orthanc에서 DICOM 이미지 가져오기
        client = OrthancClient()
        dicom_data = client.get_instance_file(instance_id)
        
        # DICOM을 PNG로 변환 (Pillow 사용)
        dicom = pydicom.dcmread(BytesIO(dicom_data))
        pixel_array = dicom.pixel_array
        
        # 정규화 (0-255)
        if pixel_array.max() > 255:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # PIL Image로 변환
        if len(pixel_array.shape) == 2:  # Grayscale
            image = Image.fromarray(pixel_array, mode='L').convert('RGB')
        else:
            image = Image.fromarray(pixel_array)
        
        # Base64 인코딩
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # AI 서비스 호출
        confidence = request.data.get('confidence', 0.25)
        iou_threshold = request.data.get('iou_threshold', 0.45)
        
        ai_request = {
            'instance_id': instance_id,
            'image_data': image_base64,
            'confidence': confidence,
            'iou_threshold': iou_threshold
        }
        
        logger.info(f"Calling mammography AI service for instance {instance_id}")
        ai_response = requests.post(
            f"{MAMMOGRAPHY_AI_SERVICE_URL}/inference",
            json=ai_request,
            timeout=60
        )
        ai_response.raise_for_status()
        
        result = ai_response.json()
        
        # 응답 구성
        response_data = {
            'success': result.get('success', False),
            'instance_id': instance_id,
            'detections': result.get('detections', []),
            'annotated_image_base64': result.get('annotated_image', ''),
            'error': result.get('error', '')
        }
        
        logger.info(f"Detection completed: {len(result.get('detections', []))} objects found")
        
        return Response(response_data)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"AI service request failed: {str(e)}")
        return Response({
            'success': False,
            'error': f'AI 서비스 연결 실패: {str(e)}'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def mammography_ai_health(request):
    """
    유방촬영술 AI 서비스 헬스 체크
    
    GET /api/mri/mammography/ai/health/
    """
    try:
        response = requests.get(f"{MAMMOGRAPHY_AI_SERVICE_URL}/health", timeout=5)
        return Response({
            'success': True,
            'service': 'mammography_ai',
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'url': MAMMOGRAPHY_AI_SERVICE_URL
        })
    except Exception as e:
        return Response({
            'success': False,
            'service': 'mammography_ai',
            'status': 'unavailable',
            'error': str(e),
            'url': MAMMOGRAPHY_AI_SERVICE_URL
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
