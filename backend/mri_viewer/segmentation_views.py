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
    Body (optional): {
        "sequence_instance_ids": [id1, id2, id3, id4]  // 4-channel DCE-MRI
    }
    """
    try:
        # Request body에서 4개 시퀀스 ID 가져오기 (없으면 단일 이미지 모드)
        sequence_ids = request.data.get('sequence_instance_ids', [instance_id])
        
        logger.info(f"🔍 MRI 세그멘테이션 시작: {len(sequence_ids)}개 시퀀스")
        logger.info(f"   Instance IDs: {sequence_ids}")
        
        # 1. Orthanc에서 DICOM 이미지들 가져오기
        client = OrthancClient()
        
        if len(sequence_ids) == 4:
            # 4-channel DCE-MRI: 4개 시퀀스를 모두 가져와서 전송
            dicom_data_list = []
            for seq_id in sequence_ids:
                dicom_data = client.get_instance_file(seq_id)
                dicom_data_list.append(dicom_data)
            
            # JSON으로 4개 시퀀스 전송
            import json
            payload = json.dumps({
                'sequences': [base64.b64encode(d).decode('utf-8') for d in dicom_data_list]
            })
            
            logger.info(f"📡 4-channel 세그멘테이션 API 호출: {SEGMENTATION_API_URL}/inference")
            
            seg_response = requests.post(
                f"{SEGMENTATION_API_URL}/inference",
                data=payload,
                headers={'Content-Type': 'application/json'},
                timeout=600
            )
        else:
            # 단일 이미지 모드 (기존 방식)
            dicom_data = client.get_instance_file(instance_id)
            
            logger.info(f"📡 단일 이미지 세그멘테이션 API 호출: {SEGMENTATION_API_URL}/inference")
            
            seg_response = requests.post(
                f"{SEGMENTATION_API_URL}/inference",
                data=dicom_data,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=600
            )
        
        seg_response.raise_for_status()
        seg_result = seg_response.json()
        
        if not seg_result.get('success'):
            raise Exception(seg_result.get('error', '세그멘테이션 실패'))
        
        # 3. 결과 반환 (마스크는 base64로 인코딩되어 있음)
        response_data = {
            'success': True,
            'instance_id': instance_id,
            'segmentation_mask_base64': seg_result.get('segmentation_mask_base64', ''),
            'tumor_pixel_count': seg_result.get('tumor_pixel_count', 0),
            'total_pixel_count': seg_result.get('total_pixel_count', 0),
            'tumor_ratio_percent': seg_result.get('tumor_ratio_percent', 0.0),
            'image_size': seg_result.get('image_size', []),
            'seg_instance_id': seg_result.get('seg_instance_id'),  # Orthanc에 저장된 세그멘테이션 Instance ID
            'saved_to_orthanc': seg_result.get('saved_to_orthanc', False),
        }
        
        logger.info(f"✅ 세그멘테이션 완료: 종양 비율 {response_data['tumor_ratio_percent']:.2f}%")
        if response_data['saved_to_orthanc']:
            logger.info(f"💾 Orthanc 저장 완료: {response_data['seg_instance_id']}")
        return Response(response_data)
        
    except requests.exceptions.Timeout:
        logger.error("⏱️ 세그멘테이션 API 타임아웃")
        return Response({
            'success': False,
            'instance_id': instance_id,
            'error': '세그멘테이션 API 타임아웃 (600초 초과)'
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

