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
import json
import uuid
from datetime import datetime
from google.cloud import storage
from .orthanc_client import OrthancClient

logger = logging.getLogger(__name__)

# 세그멘테이션 API 서버 URL (Mosec)
SEGMENTATION_API_URL = "http://localhost:5006"

# GCS 설정
GCS_BUCKET_NAME = "hospital-mri-temp-data"
GCS_TEMP_FOLDER = "mri_temp"


def upload_to_gcs(data_dict, filename=None):
    """
    데이터를 GCS에 업로드하고 Public URL 반환
    
    Args:
        data_dict: 업로드할 데이터 (dict)
        filename: 파일명 (없으면 UUID 생성)
    
    Returns:
        str: GCS Public URL
    """
    if filename is None:
        filename = f"{uuid.uuid4().hex}.json"
    
    blob_name = f"{GCS_TEMP_FOLDER}/{datetime.now().strftime('%Y%m%d')}/{filename}"
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        
        # JSON 데이터 업로드
        json_data = json.dumps(data_dict)
        blob.upload_from_string(json_data, content_type='application/json')
        
        # Public URL 생성
        blob.make_public()
        public_url = blob.public_url
        
        logger.info(f"✅ GCS 업로드 완료: {blob_name} ({len(json_data) / (1024**2):.2f} MB)")
        return public_url
        
    except Exception as e:
        logger.error(f"❌ GCS 업로드 실패: {e}", exc_info=True)
        raise


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


@api_view(['POST'])
def segment_series(request, series_id):
    """
    시리즈 전체를 3D 세그멘테이션하고 Orthanc에 저장 (4-channel, 96 슬라이스)
    
    POST /api/mri/segmentation/series/<series_id>/segment/
    Body (required): {
        "sequence_series_ids": [series1_id, series2_id, series3_id, series4_id]  // 4-channel 필수
    }
    """
    try:
        logger.info(f"🔍 시리즈 3D 세그멘테이션 시작: series_id={series_id}")
        
        client = OrthancClient()
        
        # 요청 body에서 4개 시퀀스 ID 가져오기 (필수)
        sequence_series_ids = request.data.get("sequence_series_ids", [])
        
        # 4개 시리즈 필수 체크
        if len(sequence_series_ids) != 4:
            return Response({
                "success": False,
                "error": "4개 시리즈가 모두 필요합니다. DCE-MRI 세그멘테이션을 위해서는 "
                         "Seq0, Seq1, Seq2, SeqLast 시리즈가 모두 선택되어야 합니다."
            }, status=400)
        
        # 현재 시리즈 정보 가져오기 (UI에서 선택된 메인 시리즈)
        main_series_info = client.get(f"/series/{series_id}")
        main_instances = main_series_info.get("Instances", [])
        total_slices = len(main_instances)
        
        if total_slices < 96:
            return Response({
                "success": False,
                "error": f"슬라이스 수가 부족합니다 (최소 96개 필요, 현재 {total_slices}개)"
            }, status=400)
        
        # 세그멘테이션을 위한 고유 Series UID 생성
        from pydicom.uid import generate_uid
        seg_series_uid = generate_uid()
        
        logger.info(f"🚀 세그멘테이션 Series UID: {seg_series_uid}")
        
        # 중앙 부분에서 96개 슬라이스 선택
        start_idx = (total_slices - 96) // 2
        end_idx = start_idx + 96
        
        logger.info(f"📍 슬라이스 선택: {start_idx}~{end_idx-1}번 (중앙 96개)")
        
        # 4개 시퀀스에서 각각 96개 슬라이스 수집
        sequences_3d = []  # [4][96] 형태 (각 요소는 base64 인코딩된 DICOM)
        
        for seq_idx, current_seq_series_id in enumerate(sequence_series_ids):
            seq_info = client.get(f"/series/{current_seq_series_id}")
            seq_instances = seq_info.get("Instances", [])
            
            if len(seq_instances) < 96:
                return Response({
                    "success": False,
                    "error": f"시퀀스 {current_seq_series_id}의 슬라이스가 부족합니다 (최소 96개 필요)"
                }, status=400)
            
            # 같은 범위에서 96개 선택
            selected_instances = seq_instances[start_idx:end_idx]
            
            # 각 슬라이스의 DICOM 데이터 수집 (base64 인코딩)
            slices_data = []
            for instance_id in selected_instances:
                dicom_data = client.get_instance_file(instance_id)
                slices_data.append(base64.b64encode(dicom_data).decode("utf-8"))
            
            sequences_3d.append(slices_data)  # [96] 크기의 리스트 추가
            logger.info(f"✅ 시퀀스 {seq_idx+1}/4 수집 완료: {len(slices_data)}개 슬라이스")
        
        # 1. DICOM 데이터를 GCS에 업로드
        logger.info("📤 DICOM 데이터를 GCS에 업로드 중...")
        
        gcs_payload = {
            "sequences_3d": sequences_3d,  # [4][96] 형태, 각 요소는 base64 인코딩된 DICOM
            "seg_series_uid": seg_series_uid,
            "original_series_id": series_id,
            "start_instance_number": start_idx + 1
        }
        
        gcs_url = upload_to_gcs(gcs_payload, f"mri_{seg_series_uid}.json")
        
        # 2. Mosec에는 GCS URL만 전송 (작은 payload)
        logger.info(f"📡 Mosec으로 GCS URL 전송 중...")
        
        seg_response = requests.post(
            f"{SEGMENTATION_API_URL}/inference",
            json={
                "gcs_url": gcs_url,
                "seg_series_uid": seg_series_uid,
                "original_series_id": series_id,
            },
            timeout=600  # 10분
        )
        
        seg_response.raise_for_status()
        result = seg_response.json()
        
        logger.info(f"✅ 세그멘테이션 완료!")
        
        return Response({
            'success': True,
            'series_id': series_id,
            'total_slices': 96,
            'seg_instance_id': result.get('seg_instance_id'),
            'tumor_ratio_percent': result.get('tumor_ratio_percent', 0),
            'saved_to_orthanc': result.get('saved_to_orthanc', False)
        })
        
    except Exception as e:
        logger.error(f"❌ 시리즈 세그멘테이션 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'series_id': series_id,
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

