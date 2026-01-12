"""
병리 이미지 분류 API 뷰
"""
import logging
import json
import base64
import requests
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

logger = logging.getLogger(__name__)

# Mosec 서비스 URL
PATHOLOGY_MOSEC_URL = "http://localhost:5008/inference"


@api_view(['POST'])
def pathology_ai_analysis(request):
    """
    병리 이미지 AI 분석 (CLAM)
    
    Request Body:
        {
            "svs_file_base64": "base64 encoded SVS file"
        }
    
    Response:
        {
            "success": true,
            "class_id": 1,
            "class_name": "Tumor",
            "confidence": 0.95,
            "probabilities": {
                "Normal": 0.05,
                "Tumor": 0.95
            },
            "num_patches": 856,
            "top_attention_patches": [123, 456, 789, ...]
        }
    """
    try:
        # 요청 데이터 파싱
        svs_file_base64 = request.data.get('svs_file_base64')
        
        if not svs_file_base64:
            return Response(
                {'error': 'svs_file_base64가 필요합니다'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"📥 병리 이미지 분석 요청")
        logger.info(f"📊 SVS 파일 크기: {len(svs_file_base64)} bytes (base64)")
        
        # Mosec 서비스 호출
        payload = {
            "svs_file_base64": svs_file_base64
        }
        
        logger.info(f"🚀 Mosec 서비스 호출: {PATHOLOGY_MOSEC_URL}")
        
        response = requests.post(
            PATHOLOGY_MOSEC_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=300  # 5분 타임아웃 (WSI 처리 시간 고려)
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Mosec 서비스 오류: {response.status_code} - {response.text}")
            return Response(
                {'error': f'Mosec 서비스 오류: {response.status_code}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # 응답 파싱
        mosec_result = response.json()
        logger.info(f"📥 Mosec 응답 내용: {mosec_result}")
        
        # 결과 추출
        if 'results' in mosec_result:
            result = mosec_result['results']
        else:
            result = mosec_result
        
        logger.info(f"✅ 병리 이미지 분석 완료: {result.get('class_name', 'Unknown')}")
        
        return Response(result, status=status.HTTP_200_OK)
        
    except requests.exceptions.Timeout:
        logger.error(f"❌ Mosec 서비스 타임아웃")
        return Response(
            {'error': 'AI 분석 타임아웃 (5분 초과)'},
            status=status.HTTP_504_GATEWAY_TIMEOUT
        )
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ Mosec 서비스 연결 실패")
        return Response(
            {'error': 'Mosec 서비스에 연결할 수 없습니다'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    except Exception as e:
        logger.error(f"❌ 병리 이미지 분석 오류: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def pathology_ai_health(request):
    """병리 AI 서비스 헬스 체크"""
    try:
        response = requests.get(
            "http://localhost:5008/",
            timeout=5
        )
        return Response({
            'status': 'healthy' if response.status_code == 200 else 'unhealthy',
            'mosec_status_code': response.status_code
        })
    except Exception as e:
        return Response({
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

