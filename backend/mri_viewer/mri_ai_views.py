"""
MRI AI 분석 API Views
Flask ML Service를 통해 MRI 모델 실행
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
def mri_ai_analysis(request, patient_id):
    """
    MRI 이미지에 대해 pCR 예측 분석 실행 (Flask ML Service를 통해)
    
    POST /api/mri/patients/<patient_id>/analyze/
    
    Request body:
    {
        "clinical_data": {  # optional
            "age": 50,
            "tumor_subtype": "luminal"
        }
    }
    
    Response:
    {
        "success": true,
        "pCR_probability": 0.75,
        "prediction": "pCR",
        "tumor_voxels": 12345,
        "clinical": {...}
    }
    """
    try:
        logger.info(f"Starting MRI analysis for patient {patient_id} via Flask ML Service")
        
        # 임상 정보 (요청에서 받거나, 기본값 사용)
        clinical_data = request.data.get('clinical_data', {})
        
        # Flask ML Service 호출 (환자 ID 전달)
        logger.info(f"Calling Flask ML Service: {ML_SERVICE_URL}/mri/analyze")
        flask_response = requests.post(
            f"{ML_SERVICE_URL}/mri/analyze",
            json={
                'patient_id': patient_id,
                'clinical_data': clinical_data
            },
            timeout=600  # 10분 타임아웃 (MRI 분석은 시간이 오래 걸릴 수 있음)
        )
        
        flask_response.raise_for_status()
        result = flask_response.json()
        
        # Flask 응답을 Django 응답 형식으로 변환
        return Response({
            'success': result.get('success', False),
            'patient_id': patient_id,
            'pCR_probability': result.get('pCR_probability'),
            'prediction': result.get('prediction'),
            'tumor_voxels': result.get('tumor_voxels'),
            'clinical': result.get('clinical', {}),
            'error': result.get('error')
        }, status=status.HTTP_200_OK if result.get('success') else status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ Flask ML Service request timed out for patient {patient_id}")
        return Response({
            'success': False,
            'error': 'AI 서비스 요청 시간 초과',
            'details': 'MRI 분석이 지정된 시간 내에 완료되지 않았습니다.'
        }, status=status.HTTP_504_GATEWAY_TIMEOUT)
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Flask ML Service request failed for patient {patient_id}: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': f'AI 서비스 요청 실패: {str(e)}',
            'details': 'Flask ML 서비스와 통신 중 오류가 발생했습니다.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        logger.error(f"❌ MRI analysis failed in Django view for patient {patient_id}: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e),
            'details': 'Django 내부 처리 중 오류가 발생했습니다.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def mri_ai_health(request):
    """
    MRI AI 서비스 헬스 체크 (Flask ML Service 호출)
    """
    try:
        flask_health_response = requests.get(f"{ML_SERVICE_URL}/health", timeout=10)
        flask_health_response.raise_for_status()
        health_data = flask_health_response.json()
        
        return Response({
            'success': True,
            'service': 'mri_ai',
            'status': 'healthy',
            'flask_service': ML_SERVICE_URL,
            'mri_models_available': health_data.get('mri_models_available', False)
        })
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Flask ML Service health check failed: {str(e)}")
        return Response({
            'success': False,
            'service': 'mri_ai',
            'status': 'unavailable',
            'error': str(e),
            'flask_service': ML_SERVICE_URL
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

