"""
MRI 세그멘테이션 API Views (MAMA_MIA_DELIVERY_PKG 파이프라인 사용)
- Orthanc 연동: 기존 시스템 로직 유지
- 추론: 새로운 MAMA_MIA 파이프라인 사용
- 연구실 컴퓨터 추론: 로컬 환경에서 추론 실행 가능
"""
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import AllowAny
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
import requests
import io
import logging
import os
import base64
import numpy as np
import pydicom
import tempfile
import shutil
import json
from pathlib import Path
from .orthanc_client import OrthancClient
import sys

# 새로운 MAMA_MIA 세그멘테이션 모듈 import (지연 로드로 변경)
# Django 시작 시 import 오류 방지를 위해 함수 내부에서 import

logger = logging.getLogger(__name__)

# Orthanc 설정
ORTHANC_URL = os.getenv('ORTHANC_URL', 'http://34.42.223.43:8042')
ORTHANC_USER = os.getenv('ORTHANC_USER', 'admin')
ORTHANC_PASSWORD = os.getenv('ORTHANC_PASSWORD', 'admin123')

# 모델 경로 (우선순위: src/best_model.pth -> checkpoints/best_model.pth)
MODEL_PATH = Path(__file__).parent.parent / "mri_segmentation" / "src" / "best_model.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).parent.parent / "mri_segmentation" / "checkpoints" / "best_model.pth"
if not MODEL_PATH.exists():
    logger.warning(f"Model file not found at expected locations. Searched: {MODEL_PATH}")

# 전역 추론 파이프라인 (한 번만 로드)
_pipeline = None

def get_pipeline():
    """추론 파이프라인 싱글톤 (지연 로드)"""
    global _pipeline
    if _pipeline is None:
        # 지연 import로 Django 시작 시 오류 방지
        sys.path.insert(0, str(Path(__file__).parent.parent / "mri_segmentation" / "src"))
        from inference_pipeline import SegmentationInferencePipeline
        
        logger.info(f"Loading segmentation model from: {MODEL_PATH}")
        
        # GPU 사용 가능 여부 확인
        import torch
        device = "cpu"  # 기본값은 CPU
        if os.getenv('USE_GPU', 'false').lower() == 'true':
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA GPU for inference")
            else:
                logger.warning("GPU requested but not available. Using CPU instead.")
        else:
            logger.info("Using CPU for inference (set USE_GPU=true to enable GPU)")
        
        _pipeline = SegmentationInferencePipeline(
            model_path=str(MODEL_PATH),
            device=device,
            threshold=0.5
        )
        logger.info("Model loaded successfully!")
    return _pipeline


# CSRF 체크를 건너뛰는 커스텀 인증 클래스
class CSRFExemptSessionAuthentication(SessionAuthentication):
    def enforce_csrf(self, request):
        return  # CSRF 체크를 건너뜀


@api_view(['POST'])
@authentication_classes([CSRFExemptSessionAuthentication])
@permission_classes([AllowAny])
def mri_segmentation(request, instance_id):
    """
    MRI 세그멘테이션 실행 및 Orthanc에 저장 (단일 인스턴스 또는 4채널)
    
    POST /api/mri/segmentation/instances/<instance_id>/segment/
    """
    try:
        # Request body에서 4개 시퀀스 ID 가져오기 (없으면 단일 이미지 모드)
        sequence_ids = request.data.get('sequence_instance_ids', [instance_id])
        
        logger.info(f"🔍 MRI 세그멘테이션 시작: {len(sequence_ids)}개 시퀀스")
        logger.info(f"   Instance IDs: {sequence_ids}")
        
        # 4채널인 경우 perform_segment_series_logic 사용
        if len(sequence_ids) == 4:
            # 각 인스턴스의 시리즈 ID 찾기
            client = OrthancClient()
            sequence_series_ids = []
            for inst_id in sequence_ids:
                inst_info = client.get_instance_info(inst_id)
                parent_series = inst_info.get('ParentSeries')
                if parent_series:
                    sequence_series_ids.append(parent_series)
            
            if len(sequence_series_ids) == 4:
                return _perform_segment_series_logic(request, sequence_series_ids[0], sequence_series_ids)
            else:
                return Response({
                    'success': False,
                    'error': '4개 시퀀스의 시리즈 ID를 찾을 수 없습니다.'
                }, status=400)
        else:
            # 단일 이미지 모드는 아직 지원하지 않음
            return Response({
                'success': False,
                'error': '단일 이미지 모드는 지원하지 않습니다. 4채널 DCE-MRI만 지원합니다.'
            }, status=400)
            
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
        # 모델 파일 존재 여부 확인
        model_file_exists = MODEL_PATH.exists()
        
        # 모델 로드 가능 여부 확인
        model_loaded = False
        error_msg = None
        try:
            pipeline = get_pipeline()
            model_loaded = pipeline is not None
        except Exception as e:
            logger.warning(f"모델 로드 실패: {e}")
            error_msg = str(e)
            model_loaded = False
        
        return Response({
            'success': True,
            'status': 'healthy' if model_loaded else 'model_not_loaded',
            'service': 'New Segmentation Pipeline',
            'model_loaded': model_loaded,
            'model_file_exists': model_file_exists,
            'model_path': str(MODEL_PATH),
            'orthanc_url': ORTHANC_URL,
            'error': error_msg if error_msg else None
        })
    except Exception as e:
        return Response({
            'success': False,
            'status': 'unavailable',
            'error': str(e)
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(['POST'])
@authentication_classes([CSRFExemptSessionAuthentication])
@permission_classes([AllowAny])
def segment_series(request, series_id):
    """
    시리즈 전체를 3D 세그멘테이션하고 Orthanc에 저장
    
    POST /api/mri/segmentation/series/<series_id>/segment/
    """
    sequence_series_ids = request.data.get("sequence_series_ids", [])
    if len(sequence_series_ids) != 4:
        return Response({
            "success": False,
            "error": "4개 시리즈가 모두 필요합니다."
        }, status=400)
    
    return _perform_segment_series_logic(request, series_id, sequence_series_ids)


def _perform_segment_series_logic(request, series_id, sequence_series_ids):
    """
    시리즈 세그멘테이션의 핵심 로직 (내부 helper 함수)
    """
    try:
        logger.info(f"🔍 시리즈 3D 세그멘테이션 시작: series_id={series_id}")
        
        # 연구실 컴퓨터 사용 여부 확인
        use_local = request.query_params.get('use_local', '').lower() == 'true'
        force_gcp = request.query_params.get('force_gcp', '').lower() == 'true'
        
        if not use_local and not force_gcp:
            use_local = os.getenv('USE_LOCAL_INFERENCE', 'false').lower() == 'true'
        
        # 연구실 컴퓨터 워커 사용 시
        if use_local and not force_gcp:
            logger.info("🏠 연구실 컴퓨터 워커를 통해 추론 요청 생성")
            return _create_local_inference_request(request, series_id, sequence_series_ids)
        
        # GCP에서 직접 실행
        logger.info("☁️ GCP 서버에서 직접 추론 실행")
        
        client = OrthancClient()
        temp_dir = tempfile.mkdtemp(prefix="mri_seg_")
        
        try:
            # 1. Orthanc에서 4개 시퀀스의 DICOM 파일 다운로드 및 저장
            for seq_idx, seq_series_id in enumerate(sequence_series_ids):
                seq_info = client.get(f"/series/{seq_series_id}")
                seq_instances = seq_info.get("Instances", [])
                
                if len(seq_instances) == 0:
                    return Response({
                        "success": False,
                        "error": f"시퀀스 {seq_idx+1}에 슬라이스가 없습니다."
                    }, status=400)
                
                seq_dir = Path(temp_dir) / f"seq_{seq_idx:02d}"
                seq_dir.mkdir(parents=True, exist_ok=True)
                
                for inst_idx, instance_id in enumerate(seq_instances):
                    dicom_bytes = client.get_instance_file(instance_id)
                    dicom_path = seq_dir / f"slice_{inst_idx:04d}.dcm"
                    with open(dicom_path, 'wb') as f:
                        f.write(dicom_bytes)
                
                logger.info(f"✅ 시퀀스 {seq_idx+1}/4: {len(seq_instances)}개 슬라이스 저장 완료")
            
            # 2. MAMA-MIA 모델 로드 (싱글톤 사용)
            pipeline = get_pipeline()
            if pipeline is None:
                return Response({
                    "success": False,
                    "error": "세그멘테이션 모델을 로드할 수 없습니다."
                }, status=500)
            
            # 3. 추론 실행
            seg_dicom_path = Path(temp_dir) / "segmentation.dcm"
            result = pipeline.predict(
                image_path=temp_dir,
                output_path=str(seg_dicom_path),
                output_format="dicom"
            )
            
            logger.info(f"✅ 추론 완료: tumor_detected={result['tumor_detected']}")
            
            # 4. DICOM SEG를 Orthanc에 업로드
            if not seg_dicom_path.exists():
                raise Exception("세그멘테이션 결과 파일(DICOM SEG)이 생성되지 않았습니다.")
                
            with open(seg_dicom_path, 'rb') as f:
                seg_dicom_bytes = f.read()
            
            upload_result = client.upload_dicom(seg_dicom_bytes)
            seg_instance_id = upload_result.get('ID')
            
            logger.info(f"✅ Orthanc 업로드 완료: {seg_instance_id}")
            
            return Response({
                'success': True,
                'series_id': series_id,
                'tumor_detected': result['tumor_detected'],
                'tumor_volume_voxels': result['tumor_volume_voxels'],
                'seg_instance_id': seg_instance_id,
                'saved_to_orthanc': True
            })
            
        finally:
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
                logger.info("🧹 임시 파일 정리 완료")
    
    except Exception as e:
        logger.error(f"❌ 세그멘테이션 로직 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'series_id': series_id,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_segmentation_frames(request, seg_instance_id):
    """
    DICOM SEG 파일에서 모든 프레임을 추출하여 반환
    """
    try:
        logger.info(f"🔍 DICOM SEG 프레임 추출 시작: {seg_instance_id}")
        client = OrthancClient()
        seg_dicom_bytes = client.get_instance_file(seg_instance_id)
        
        dicom_data = io.BytesIO(seg_dicom_bytes)
        ds = pydicom.dcmread(dicom_data, force=True)
        
        num_frames = getattr(ds, 'NumberOfFrames', 1)
        rows = ds.Rows
        cols = ds.Columns
        
        try:
            pixel_array = ds.pixel_array
            if pixel_array.ndim == 2:
                pixel_array = pixel_array[np.newaxis, ...]
        except:
            pixel_data = np.frombuffer(ds.PixelData, dtype=np.uint8)
            if ds.BitsAllocated == 1:
                pixel_array = np.unpackbits(pixel_data).reshape(num_frames, rows, cols)
            else:
                pixel_array = pixel_data.reshape(num_frames, rows, cols)
        
        frames = []
        for i in range(num_frames):
            frame_data = (pixel_array[i] > 0).astype(np.uint8) * 255
            from PIL import Image
            img = Image.fromarray(frame_data, mode='L')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            frames.append({
                "index": i,
                "mask_base64": mask_base64
            })
        
        return Response({
            "success": True,
            "num_frames": len(frames),
            "frames": frames
        })
        
    except Exception as e:
        logger.error(f"❌ 프레임 추출 실패: {str(e)}", exc_info=True)
        return Response({
            "success": False,
            "error": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============================================================
# 연구실 컴퓨터 추론 요청 API
# ============================================================

REQUEST_DIR = Path(os.getenv('INFERENCE_REQUEST_DIR', '/tmp/mri_inference_requests'))

def _create_local_inference_request(request, series_id, sequence_series_ids):
    """
    연구실 컴퓨터에서 추론 실행 요청 생성 (내부 함수)
    """
    try:
        REQUEST_DIR.mkdir(exist_ok=True, parents=True)
        
        request_data = {
            'series_ids': sequence_series_ids,
            'main_series_id': series_id,
            'requested_at': timezone.now().isoformat(),
            'status': 'pending',
            'requested_by': getattr(request.user, 'username', 'anonymous') if hasattr(request, 'user') and hasattr(request.user, 'is_authenticated') and request.user.is_authenticated else 'anonymous'
        }
        
        timestamp = int(timezone.now().timestamp() * 1000)
        request_id = f"{series_id}_{timestamp}"
        request_file = REQUEST_DIR / f"{request_id}.json"
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 요청 생성: {request_file.name}")
        
        import time
        max_wait_time = 600  # 10분 (다운로드 시간이 길 수 있음)
        check_interval = 2
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            time.sleep(check_interval)
            elapsed_time += check_interval
            
            try:
                with open(request_file, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
                
                current_status = current_data.get('status')
                if current_status == 'completed':
                    result = current_data.get('result', {})
                    return Response({
                        'success': True,
                        'series_id': series_id,
                        'request_id': request_id,
                        'tumor_detected': result.get('tumor_detected'),
                        'tumor_volume_voxels': result.get('tumor_volume_voxels'),
                        'seg_instance_id': result.get('seg_instance_id'),
                        'elapsed_time_seconds': result.get('elapsed_time_seconds'),
                        'saved_to_orthanc': True,
                        'processed_by': 'local_worker'
                    })
                elif current_status == 'failed':
                    result = current_data.get('result', {})
                    return Response({
                        'success': False,
                        'error': result.get('error', '알 수 없는 오류'),
                        'request_id': request_id
                    }, status=500)
            except:
                pass
        
        return Response({
            'success': False,
            'error': f'요청 처리 시간 초과 (최대 {max_wait_time}초)',
            'request_id': request_id
        }, status=504)
        
    except Exception as e:
        logger.error(f"❌ 추론 요청 생성 실패: {str(e)}", exc_info=True)
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['POST'])
def request_local_inference(request, series_id):
    """
    연구실 컴퓨터에서 추론 실행 요청
    """
    sequence_series_ids = request.data.get("sequence_series_ids", [])
    if len(sequence_series_ids) != 4:
        return Response({'success': False, 'error': '4개 시리즈 필요'}, status=400)
    return _create_local_inference_request(request, series_id, sequence_series_ids)


@api_view(['GET'])
def check_inference_status(request, request_id):
    """
    추론 요청 상태 확인
    """
    try:
        request_files = list(REQUEST_DIR.glob(f"{request_id}.json"))
        if not request_files:
            return Response({'success': False, 'error': '요청 없음'}, status=404)
        
        with open(request_files[0], 'r', encoding='utf-8') as f:
            request_data = json.load(f)
        
        return Response({'success': True, 'request_id': request_id, 'status': request_data.get('status'), 'result': request_data.get('result')})
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['GET'])
def list_inference_requests(request):
    """
    추론 요청 목록 조회
    """
    try:
        request_files = sorted(REQUEST_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        requests_list = []
        for rf in request_files[:50]:
            with open(rf, 'r', encoding='utf-8') as f:
                d = json.load(f)
                requests_list.append({'request_id': rf.stem, 'status': d.get('status'), 'requested_at': d.get('requested_at')})
        return Response({'success': True, 'requests': requests_list})
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['GET'])
def get_pending_requests(request):
    """
    워커용: 대기 중인 요청 조회
    """
    try:
        request_files = sorted(REQUEST_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime)
        pending = []
        for rf in request_files:
            with open(rf, 'r', encoding='utf-8') as f:
                d = json.load(f)
                if d.get('status') == 'pending':
                    pending.append({'request_id': rf.stem, 'series_ids': d.get('series_ids'), 'main_series_id': d.get('main_series_id')})
        return Response({'success': True, 'requests': pending})
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['GET'])
def get_pending_inference(request):
    """
    조원님 워커 호환용
    """
    try:
        request_files = sorted(REQUEST_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime)
        for rf in request_files:
            with open(rf, 'r', encoding='utf-8') as f:
                d = json.load(f)
                if d.get('status') == 'pending':
                    d['status'] = 'processing'
                    with open(rf, 'w', encoding='utf-8') as f2:
                        json.dump(d, f2, indent=2, ensure_ascii=False)
                    return Response({'id': rf.stem, 'series_id': d.get('main_series_id'), 'series_ids': d.get('series_ids', [])})
        return Response({'id': None})
    except Exception as e:
        return Response({'id': None, 'error': str(e)}, status=500)


@api_view(['POST'])
def complete_inference(request, request_id):
    """
    조원님 워커 호환용
    """
    try:
        request_file = REQUEST_DIR / f"{request_id}.json"
        if not request_file.exists(): return Response({'success': False}, status=404)
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        d['status'] = 'completed' if request.data.get('success') else 'failed'
        d['result'] = request.data
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        return Response({'success': True})
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)


@api_view(['POST'])
@authentication_classes([CSRFExemptSessionAuthentication])
@permission_classes([AllowAny])
def complete_inference_request(request, request_id):
    return complete_inference(request, request_id)


@api_view(['POST'])
def update_request_status(request, request_id):
    try:
        request_file = REQUEST_DIR / f"{request_id}.json"
        if not request_file.exists(): return Response({'success': False}, status=404)
        with open(request_file, 'r', encoding='utf-8') as f:
            d = json.load(f)
        d['status'] = request.data.get('status', d['status'])
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        return Response({'success': True})
    except Exception as e:
        return Response({'success': False}, status=500)
