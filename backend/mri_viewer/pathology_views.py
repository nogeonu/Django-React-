"""
병리 이미지 분류 API 뷰
"""
import os
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
            "instance_id": "Orthanc instance ID"
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
        # Orthanc 설정
        ORTHANC_URL = os.getenv('ORTHANC_URL', 'http://localhost:8042')
        ORTHANC_USERNAME = os.getenv('ORTHANC_USERNAME', 'admin')
        ORTHANC_PASSWORD = os.getenv('ORTHANC_PASSWORD', 'admin123')
        
        # 요청 데이터 파싱
        instance_id = request.data.get('instance_id')
        
        if not instance_id:
            return Response(
                {'error': 'instance_id가 필요합니다'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        logger.info(f"📥 병리 이미지 분석 요청: instance_id={instance_id}")
        
        # Orthanc에서 DICOM 메타데이터 조회 (원본 SVS 경로 확인)
        logger.info(f"📥 Orthanc에서 DICOM 메타데이터 조회 중...")
        metadata_response = requests.get(
            f"{ORTHANC_URL}/instances/{instance_id}/tags?simplify",
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
            timeout=30
        )
        
        if metadata_response.status_code != 200:
            logger.error(f"❌ Orthanc 메타데이터 조회 실패: {metadata_response.status_code}")
            return Response(
                {'error': f'Orthanc에서 메타데이터를 가져올 수 없습니다: {metadata_response.status_code}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        metadata = metadata_response.json()
        
        # Private Tag에서 원본 SVS 경로 추출 (0011,1001)
        original_svs_path = metadata.get('0011,1001')
        
        if not original_svs_path or not os.path.exists(original_svs_path):
            logger.error(f"❌ 원본 SVS 파일을 찾을 수 없습니다: {original_svs_path}")
            return Response(
                {'error': '원본 SVS 파일을 찾을 수 없습니다. 이미지를 다시 업로드해주세요.'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        logger.info(f"✅ 원본 SVS 파일 경로: {original_svs_path}")
        
        # 원본 SVS 파일 읽기
        with open(original_svs_path, 'rb') as f:
            svs_bytes = f.read()
        
        logger.info(f"✅ SVS 파일 읽기 완료: {len(svs_bytes)} bytes")
        
        # Base64 인코딩
        svs_file_base64 = base64.b64encode(svs_bytes).decode('utf-8')
        logger.info(f"📊 Base64 인코딩 완료: {len(svs_file_base64)} bytes")
        
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

