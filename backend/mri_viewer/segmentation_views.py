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


@api_view(['POST'])
def segment_series(request, series_id):
    """
    시리즈 전체를 세그멘테이션하고 Orthanc에 저장
    
    POST /api/mri/segmentation/series/<series_id>/segment/
    Body (optional): {
        "sequence_series_ids": [series1_id, series2_id, series3_id, series4_id]  // 4-channel 모드
    }
    """
    try:
        logger.info(f"🔍 시리즈 전체 세그멘테이션 시작: series_id={series_id}")
        
        # 1. Orthanc에서 시리즈의 모든 인스턴스 가져오기
        client = OrthancClient()
        series_info = client.get(f'/series/{series_id}')
        instance_ids = series_info.get('Instances', [])
        
        if not instance_ids:
            raise Exception('시리즈에 이미지가 없습니다')
        
        logger.info(f"📊 총 {len(instance_ids)}개 슬라이스 세그멘테이션 시작")
        
        # 2. 세그멘테이션 시리즈를 위한 고유 Series Instance UID 생성
        from pydicom.uid import generate_uid
        seg_series_uid = generate_uid()
        logger.info(f"🆔 세그멘테이션 Series UID: {seg_series_uid}")
        
        # 3. 4-channel 모드 확인
        sequence_series_ids = request.data.get('sequence_series_ids', [])
        is_4channel = len(sequence_series_ids) == 4
        
        # 4. 병렬 처리로 모든 슬라이스 세그멘테이션 (10개씩 동시 처리)
        logger.info(f"🚀 병렬 세그멘테이션 시작: {len(instance_ids)}개 슬라이스")
        
        import concurrent.futures
        import threading
        
        results = []
        seg_instance_ids = []
        results_lock = threading.Lock()
        
        def process_slice(idx, instance_id):
            """개별 슬라이스 처리"""
            try:
                if is_4channel:
                    # 4개 시리즈에서 같은 인덱스의 인스턴스 수집
                    sequence_instance_ids = []
                    for seq_series_id in sequence_series_ids:
                        seq_info = client.get(f'/series/{seq_series_id}')
                        seq_instances = seq_info.get('Instances', [])
                        if idx < len(seq_instances):
                            sequence_instance_ids.append(seq_instances[idx])
                    
                    if len(sequence_instance_ids) != 4:
                        logger.warning(f"  ⚠️ 슬라이스 {idx}: 4개 시퀀스를 찾을 수 없음, 스킵")
                        return None
                    
                    # 4-channel 세그멘테이션
                    dicom_data_list = []
                    for seq_id in sequence_instance_ids:
                        dicom_data = client.get_instance_file(seq_id)
                        dicom_data_list.append(dicom_data)
                    
                    payload = {
                        'sequences': [base64.b64encode(d).decode('utf-8') for d in dicom_data_list],
                        'seg_series_uid': seg_series_uid,
                        'instance_number': idx + 1
                    }
                else:
                    # 단일 이미지 세그멘테이션
                    dicom_data = client.get_instance_file(instance_id)
                    
                    payload = {
                        'dicom_data': base64.b64encode(dicom_data).decode('utf-8'),
                        'seg_series_uid': seg_series_uid,
                        'instance_number': idx + 1
                    }
                
                # Mosec으로 전송
                seg_response = requests.post(
                    f"{SEGMENTATION_API_URL}/inference",
                    json=payload,
                    timeout=60
                )
                
                seg_response.raise_for_status()
                seg_result = seg_response.json()
                
                if seg_result.get('success'):
                    result = {
                        'instance_id': instance_id,
                        'slice_index': idx,
                        'tumor_ratio_percent': seg_result.get('tumor_ratio_percent', 0),
                        'seg_instance_id': seg_result.get('seg_instance_id')
                    }
                    
                    with results_lock:
                        results.append(result)
                        if seg_result.get('seg_instance_id'):
                            seg_instance_ids.append(seg_result.get('seg_instance_id'))
                    
                    if (idx + 1) % 10 == 0:
                        logger.info(f"  진행: {idx + 1}/{len(instance_ids)}")
                    
                    return result
                else:
                    logger.error(f"  ❌ 슬라이스 {idx} 실패: {seg_result.get('error')}")
                    return None
                    
            except Exception as e:
                logger.error(f"  ❌ 슬라이스 {idx} 처리 실패: {e}")
                with results_lock:
                    results.append({
                        'instance_id': instance_id,
                        'slice_index': idx,
                        'error': str(e)
                    })
                return None
        
        # 5. ThreadPoolExecutor로 병렬 처리 (10개씩 동시)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(process_slice, idx, instance_id)
                for idx, instance_id in enumerate(instance_ids)
            ]
            
            # 완료 대기
            concurrent.futures.wait(futures)
        
        # 6. 결과 반환
        logger.info(f"✅ 병렬 세그멘테이션 완료: {len(seg_instance_ids)}/{len(instance_ids)} 성공")
        
        return Response({
            'success': True,
            'series_id': series_id,
            'total_slices': len(instance_ids),
            'processed_slices': len(results),
            'successful_slices': len(seg_instance_ids),
            'results': results,
            'seg_instance_ids': seg_instance_ids,
            'is_4channel': is_4channel
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

