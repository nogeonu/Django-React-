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
def mammography_ai_analysis(request):
    """
    맘모그래피 4장 이미지 AI 분석
    
    POST /api/mri/mammography/analyze/
    Body: {
        "instance_ids": ["id1", "id2", "id3", "id4"]
    }
    
    Returns: {
        "success": true,
        "results": [
            {
                "instance_id": "...",
                "view": "L-CC",
                "predicted_class": 0,
                "probability": 0.95,
                "all_probabilities": [0.95, 0.03, 0.01, 0.01]
            },
            ...
        ]
    }
    """
    try:
        instance_ids = request.data.get('instance_ids')
        
        if not instance_ids or not isinstance(instance_ids, list):
            return Response({
                'success': False,
                'error': 'instance_ids 배열이 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        if len(instance_ids) != 4:
            return Response({
                'success': False,
                'error': '맘모그래피는 4장의 이미지가 필요합니다 (L-CC, L-MLO, R-CC, R-MLO).'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        logger.info(f"📊 맘모그래피 4장 분석 시작: {instance_ids}")
        
        # 1. Orthanc에서 4개 DICOM 파일 다운로드 + Base64 인코딩
        client = OrthancClient()
        dicom_data_list = []
        
        for instance_id in instance_ids:
            dicom_data = client.get_instance_file(instance_id)
            dicom_base64 = base64.b64encode(dicom_data).decode('utf-8')
            dicom_data_list.append({"dicom_data": dicom_base64})
            logger.info(f"📥 DICOM 데이터 로드: {instance_id} ({len(dicom_data)} bytes)")
        
        # 2. Mosec 서비스 호출 (배치 처리)
        logger.info(f"🚀 Mosec 서비스 호출 중... (4장 배치)")
        response = requests.post(
            f"{MAMMOGRAPHY_API_URL}/inference",
            json=dicom_data_list,
            timeout=120  # 2분 (4장 처리)
        )
        
        if response.status_code != 200:
            raise Exception(f"Mosec 서비스 오류: {response.status_code} - {response.text}")
        
        mosec_results = response.json()
        
        # 3. 결과 매핑 (뷰 정보는 DICOM 태그에서 추출)
        results = []
        
        for idx, (instance_id, mosec_result) in enumerate(zip(instance_ids, mosec_results)):
            if not mosec_result.get('success'):
                raise Exception(f"이미지 {idx+1} 분석 실패: {mosec_result.get('error', 'Unknown error')}")
            
            # Orthanc에서 인스턴스 메타데이터 가져오기
            try:
                instance_info = client.get_instance_info(instance_id)
                main_tags = instance_info.get('MainDicomTags', {})
                
                view_position = main_tags.get('ViewPosition', '')  # CC, MLO 등
                image_laterality = main_tags.get('ImageLaterality', '')  # L, R
                
                # 뷰 이름 생성
                if view_position and image_laterality:
                    view_name = f"{image_laterality}-{view_position}"  # L-CC, R-MLO 등
                else:
                    view_name = f"Image {idx+1}"
                    
                logger.info(f"📋 메타데이터: {instance_id} → {view_name} (ViewPosition={view_position}, ImageLaterality={image_laterality})")
            except Exception as e:
                logger.warning(f"⚠️ 메타데이터 로드 실패: {instance_id}, 기본값 사용")
                view_name = f"Image {idx+1}"
            
            # 클래스 이름 매핑
            class_names = ['Mass', 'Calcification', 'Architectural/Asymmetry', 'Normal']
            predicted_class = mosec_result['class_id']
            
            # 모든 확률값 배열로 변환
            all_probs = [
                mosec_result['probabilities'].get('Mass', 0.0),
                mosec_result['probabilities'].get('Calcification', 0.0),
                mosec_result['probabilities'].get('Architectural/Asymmetry', 0.0),
                mosec_result['probabilities'].get('Normal', 0.0)
            ]
            
            results.append({
                'instance_id': instance_id,
                'view': view_name,
                'predicted_class': predicted_class,
                'class_name': class_names[predicted_class],
                'probability': mosec_result['confidence'],
                'all_probabilities': all_probs
            })
            
            logger.info(f"✅ {view_name}: {class_names[predicted_class]} (신뢰도: {mosec_result['confidence']:.4f})")
        
        return Response({
            'success': True,
            'results': results
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

