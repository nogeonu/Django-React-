"""
유방촬영술 AI 디텍션 API Views
Flask ML Service를 통해 YOLO 모델 실행
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import base64
import os
import logging
import requests

from .orthanc_client import OrthancClient

logger = logging.getLogger(__name__)

# Flask ML Service URL
ML_SERVICE_URL = os.getenv('ML_SERVICE_URL', 'http://localhost:5002')


@api_view(['POST'])
def mammography_ai_detection(request, instance_id):
    """
    유방촬영술 이미지에 대해 YOLO11 디텍션 실행 (Flask ML Service를 통해)
    
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
        "detections": [...],
        "annotated_image_base64": "base64_string"
    }
    """
    try:
        logger.info(f"Starting YOLO detection for instance {instance_id} via Flask ML Service")
        
        # Orthanc에서 DICOM 이미지 가져오기
        client = OrthancClient()
        dicom_data = client.get_instance_file(instance_id)
        
        # DICOM 데이터를 base64로 인코딩
        dicom_data_base64 = base64.b64encode(dicom_data).decode('utf-8')
        
        # Flask ML Service 호출
        confidence = float(request.data.get('confidence', 0.25))
        iou_threshold = float(request.data.get('iou_threshold', 0.45))
        
        logger.info(f"Calling Flask ML Service: {ML_SERVICE_URL}/mammography/detect")
        flask_response = requests.post(
            f"{ML_SERVICE_URL}/mammography/detect",
            json={
                'dicom_data': dicom_data_base64,
                'confidence': confidence,
                'iou_threshold': iou_threshold
            },
            timeout=180  # YOLO 추론은 시간이 오래 걸릴 수 있음
        )
        
        flask_response.raise_for_status()
        result = flask_response.json()
        
        # Flask 응답을 Django 응답 형식으로 변환
        response_data = {
            'success': result.get('success', False),
            'instance_id': instance_id,
            'detections': result.get('detections', []),
            'detection_count': result.get('detection_count', 0),
            'image_with_detections': f"data:image/png;base64,{result.get('annotated_image_base64', '')}" if result.get('annotated_image_base64') else "",
            'annotated_image_base64': result.get('annotated_image_base64', ''),
            'model_info': result.get('model_info', {
                'name': 'YOLO11 Mammography Detection',
                'confidence_threshold': confidence
            }),
            'error': result.get('error', '')
        }
        
        logger.info(f"✅ Detection completed: {response_data['detection_count']} objects found")
        return Response(response_data)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Flask ML Service request failed: {str(e)}")
        return Response({
            'success': False,
            'instance_id': instance_id,
            'detections': [],
            'detection_count': 0,
            'image_with_detections': '',
            'annotated_image_base64': '',
            'model_info': {
                'name': 'YOLO11 Mammography Detection',
                'confidence_threshold': 0.25
            },
            'error': f'Flask ML Service 연결 실패: {str(e)}'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
    except Exception as e:
        logger.error(f"❌ Detection failed: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'instance_id': instance_id,
            'detections': [],
            'detection_count': 0,
            'image_with_detections': '',
            'annotated_image_base64': '',
            'model_info': {
                'name': 'YOLO11 Mammography Detection',
                'confidence_threshold': 0.25
            },
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def mammography_ai_health(request):
    """
    유방촬영술 AI 서비스 헬스 체크 (Flask ML Service)
    
    GET /api/mri/mammography/ai/health/
    """
    try:
        # Flask ML Service 헬스 체크
        flask_response = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        flask_response.raise_for_status()
        flask_health = flask_response.json()
        
        return Response({
            'success': True,
            'service': 'mammography_ai',
            'status': 'healthy' if flask_health.get('yolo_loaded') else 'unavailable',
            'flask_service': ML_SERVICE_URL,
            'yolo_loaded': flask_health.get('yolo_loaded', False)
        })
    except requests.exceptions.RequestException as e:
        return Response({
            'success': False,
            'service': 'mammography_ai',
            'status': 'unavailable',
            'error': f'Flask ML Service 연결 실패: {str(e)}',
            'flask_service': ML_SERVICE_URL
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
