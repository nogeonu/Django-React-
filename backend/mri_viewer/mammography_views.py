"""
맘모그래피 AI 분석 API
Mosec 서비스 (포트 5007)를 호출하여 4-class 분류 수행
"""

import logging
import base64
import requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .orthanc_client import OrthancClient

logger = logging.getLogger(__name__)

# Mosec 맘모그래피 서비스 URL
MAMMOGRAPHY_API_URL = "http://localhost:5007"


@api_view(['POST'])
def analyze_mammography(request):
    """
    맘모그래피 이미지 AI 분석
    
    POST /api/mri/mammography/analyze/
    Body: {
        "instance_id": "orthanc_instance_id"
    }
    
    Returns: {
        "success": true,
        "instance_id": "...",
        "class_id": 0,
        "class_name": "Mass",
        "confidence": 0.95,
        "probabilities": {
            "Mass": 0.95,
            "Calcification": 0.03,
            "Architectural/Asymmetry": 0.01,
            "Normal": 0.01
        }
    }
    """
    try:
        instance_id = request.data.get('instance_id')
        
        if not instance_id:
            return Response({
                'success': False,
                'error': 'instance_id가 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        logger.info(f"📊 맘모그래피 분석 시작: {instance_id}")
        
        # 1. Orthanc에서 DICOM 파일 다운로드
        client = OrthancClient()
        dicom_data = client.get_instance_file(instance_id)
        
        # 2. Base64 인코딩
        dicom_base64 = base64.b64encode(dicom_data).decode('utf-8')
        
        logger.info(f"📥 DICOM 데이터 크기: {len(dicom_data)} bytes")
        
        # 3. Mosec 서비스 호출
        response = requests.post(
            f"{MAMMOGRAPHY_API_URL}/inference",
            json=[{"dicom_data": dicom_base64}],
            timeout=60  # 1분
        )
        
        if response.status_code != 200:
            raise Exception(f"Mosec 서비스 오류: {response.status_code} - {response.text}")
        
        result = response.json()[0]
        
        if not result.get('success'):
            raise Exception(result.get('error', 'Unknown error'))
        
        logger.info(f"✅ 분석 완료: {result['class_name']} (신뢰도: {result['confidence']:.4f})")
        
        return Response({
            'success': True,
            'instance_id': instance_id,
            'class_id': result['class_id'],
            'class_name': result['class_name'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities']
        })
        
    except requests.exceptions.Timeout:
        logger.error("❌ Mosec 서비스 타임아웃")
        return Response({
            'success': False,
            'error': 'AI 분석 서비스 타임아웃'
        }, status=status.HTTP_504_GATEWAY_TIMEOUT)
        
    except Exception as e:
        logger.error(f"❌ 맘모그래피 분석 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def mammography_health(request):
    """
    맘모그래피 AI 서비스 헬스 체크
    
    GET /api/mri/mammography/health/
    """
    try:
        response = requests.get(f"{MAMMOGRAPHY_API_URL}/", timeout=5)
        
        return Response({
            'success': True,
            'service': 'mammography',
            'status': 'healthy',
            'mosec_status_code': response.status_code
        })
        
    except Exception as e:
        logger.error(f"❌ 맘모그래피 서비스 헬스 체크 실패: {str(e)}")
        return Response({
            'success': False,
            'service': 'mammography',
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

