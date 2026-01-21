"""
MRI 세그멘테이션 API Views (MAMA_MIA_DELIVERY_PKG 파이프라인 사용)
- Orthanc 연동: 기존 시스템 로직 유지
- 추론: 새로운 MAMA_MIA 파이프라인 사용
- 연구실 컴퓨터 추론: 로컬 환경에서 추론 실행 가능
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
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


@api_view(['POST'])
def mri_segmentation(request, instance_id):
    """
    단일 인스턴스 세그멘테이션 (CSRF 면제)
    """
    # CSRF 체크 우회를 위한 커스텀 인증 클래스
    from rest_framework.authentication import SessionAuthentication
    
    class CSRFExemptSessionAuthentication(SessionAuthentication):
        def enforce_csrf(self, request):
            return  # CSRF 체크를 건너뜀
    
    # 뷰 레벨에서 인증 클래스 오버라이드
    request.authenticators = [CSRFExemptSessionAuthentication()]
    """
    MRI 세그멘테이션 실행 및 Orthanc에 저장 (단일 인스턴스 또는 4채널)
    
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
        
        # 4채널인 경우 segment_series와 동일한 로직 사용
        if len(sequence_ids) == 4:
            # 각 인스턴스의 시리즈 ID 찾기
            client = OrthancClient()
            sequence_series_ids = []
            for inst_id in sequence_ids:
                inst_info = client.get_instance_info(inst_id)
                series_id = inst_info.get('ParentSeries')
                if series_id:
                    sequence_series_ids.append(series_id)
            
            if len(sequence_series_ids) == 4:
                # segment_series 함수 호출
                request.data['sequence_series_ids'] = sequence_series_ids
                return segment_series(request, sequence_series_ids[0])
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
def segment_series(request, series_id):
    """
    시리즈 전체를 3D 세그멘테이션하고 Orthanc에 저장
    
    연구실 컴퓨터 워커가 실행 중이면 자동으로 요청 생성, 아니면 GCP에서 직접 실행
    
    POST /api/mri/segmentation/series/<series_id>/segment/
    Body (required): {
        "sequence_series_ids": [series1_id, series2_id, series3_id, series4_id]  // 4-channel 필수
    }
    Query params (optional): {
        "use_local": true/false  // 연구실 컴퓨터 사용 여부 (기본: 자동 감지)
    }
    """
    from rest_framework.authentication import SessionAuthentication
    from rest_framework.permissions import AllowAny
    import tempfile
    import shutil
    from pathlib import Path
    import sys
    
    # CSRF 체크 우회: SessionAuthentication의 enforce_csrf를 비활성화
    # 프론트엔드에서 CSRF 토큰 없이 호출 가능하도록 설정
    class CSRFExemptSessionAuthentication(SessionAuthentication):
        def enforce_csrf(self, request):
            return  # CSRF 체크를 건너뜀
    
    # 뷰 레벨에서 인증 클래스 오버라이드
    request.authenticators = [CSRFExemptSessionAuthentication()]
    
    try:
        logger.info(f"🔍 시리즈 3D 세그멘테이션 시작: series_id={series_id}")
        
        client = OrthancClient()
        
        # 요청 body에서 4개 시퀀스 ID 가져오기 (필수)
        sequence_series_ids = request.data.get("sequence_series_ids", [])
        
        # 4개 시리즈 필수 체크
        if len(sequence_series_ids) != 4:
            return Response({
                "success": False,
                "error": "4개 시리즈가 모두 필요합니다. DCE-MRI 세그멘테이션을 위해서는 Seq0, Seq1, Seq2, SeqLast 시리즈가 모두 선택되어야 합니다."
            }, status=400)
        
        # 연구실 컴퓨터 사용 여부 확인
        use_local = request.query_params.get('use_local', '').lower() == 'true'
        force_gcp = request.query_params.get('force_gcp', '').lower() == 'true'
        
        # 환경 변수로 기본값 설정 가능
        if not use_local and not force_gcp:
            use_local = os.getenv('USE_LOCAL_INFERENCE', 'false').lower() == 'true'
        
        # 연구실 컴퓨터 워커 사용 시
        if use_local and not force_gcp:
            logger.info("🏠 연구실 컴퓨터 워커를 통해 추론 요청 생성")
            # request_local_inference 로직을 인라인으로 처리 (csrf_exempt 충돌 방지)
            return _create_local_inference_request(request, series_id, sequence_series_ids)
        
        # GCP에서 직접 실행 (기존 방식)
        logger.info("☁️ GCP 서버에서 직접 추론 실행")
        
        
        # 임시 디렉토리 생성 (DICOM 파일 저장용)
        temp_dir = tempfile.mkdtemp(prefix="mri_seg_")
        reference_dicom_dir = None
        seg_dicom_path = None
        
        try:
            # 1. Orthanc에서 4개 시퀀스의 DICOM 파일 다운로드 및 저장
            logger.info("📥 Orthanc에서 4개 시퀀스의 DICOM 응답 대기/다운로드 중...")
            
            for seq_idx, seq_series_id in enumerate(sequence_series_ids):
                seq_info = client.get(f"/series/{seq_series_id}")
                seq_instances = seq_info.get("Instances", [])
                
                if len(seq_instances) == 0:
                    return Response({
                        "success": False,
                        "error": f"시퀀스 {seq_idx+1}에 슬라이스가 없습니다."
                    }, status=400)
                
                # 시퀀스별 디렉토리 생성
                seq_dir = Path(temp_dir) / f"seq_{seq_idx:02d}"
                seq_dir.mkdir(parents=True, exist_ok=True)
                
                # 각 인스턴스의 DICOM 파일 저장
                for inst_idx, instance_id in enumerate(seq_instances):
                    dicom_bytes = client.get_instance_file(instance_id)
                    dicom_path = seq_dir / f"slice_{inst_idx:04d}.dcm"
                    with open(dicom_path, 'wb') as f:
                        f.write(dicom_bytes)
                
                # 첫 번째 시퀀스를 참조 DICOM으로 사용
                if seq_idx == 0:
                    reference_dicom_dir = str(seq_dir)
                
                logger.info(f"✅ 시퀀스 {seq_idx+1}/4: {len(seq_instances)}개 슬라이스 저장 완료")
            
            # 2. MAMA-MIA 모델 로드 (싱글톤 사용)
            logger.info("🔄 MAMA-MIA 파이프라인 로드 중...")
            pipeline = get_pipeline()
            
            if pipeline is None:
                return Response({
                    "success": False,
                    "error": "세그멘테이션 모델을 로드할 수 없습니다."
                }, status=500)
            
            # 3. 추론 실행 (DICOM SEG 출력)
            logger.info("🔄 세그멘테이션 추론 중 (이 작업은 CPU에서 약 10~20초 소요될 수 있습니다)...")
            
            seg_dicom_path = Path(temp_dir) / "segmentation.dcm"
            
            # MAMA-MIA predict는 image_path가 폴더이면 내부의 시퀀스를 찾아 처리함
            result = pipeline.predict(
                image_path=temp_dir,  # 4개 seq_XX 폴더가 있는 루트 임시 폴더
                output_path=str(seg_dicom_path),
                output_format="dicom"
            )
            
            logger.info(f"✅ 추론 완료: tumor_detected={result['tumor_detected']}, volume={result['tumor_volume_voxels']} voxels")
            
            # 4. DICOM SEG를 Orthanc에 업로드
            logger.info("📤 DICOM SEG를 Orthanc에 업로드 중...")
            
            if not seg_dicom_path.exists():
                raise Exception("세그멘테이션 결과 파일(DICOM SEG)이 생성되지 않았습니다.")
                
            with open(seg_dicom_path, 'rb') as f:
                seg_dicom_bytes = f.read()
            
            upload_result = client.upload_dicom(seg_dicom_bytes)
            seg_instance_id = upload_result.get('ID')
            
            logger.info(f"✅ Orthanc 업로드 완료: {seg_instance_id}")
            
            # 5. 결과 반환
            return Response({
                'success': True,
                'series_id': series_id,
                'tumor_detected': result['tumor_detected'],
                'tumor_volume_voxels': result['tumor_volume_voxels'],
                'seg_instance_id': seg_instance_id,
                'saved_to_orthanc': True
            })
            
        except Exception as e:
            logger.error(f"❌ 작업 도중 오류 발생: {str(e)}", exc_info=True)
            raise
        finally:
            # 임시 파일 정리
            try:
                if temp_dir and Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)
                    logger.info("🧹 임시 파일 정리 완료")
            except Exception as cleanup_error:
                logger.warning(f"임시 파일 정리 중 오류: {cleanup_error}")
    
    except Exception as e:
        logger.error(f"❌ 시리즈 세그멘테이션 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'series_id': series_id,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@api_view(['GET'])
def get_segmentation_frames(request, seg_instance_id):
    """
    DICOM SEG 파일에서 모든 프레임을 추출하여 반환
    
    GET /api/mri/segmentation/instances/<seg_instance_id>/frames/
    """
    try:
        logger.info(f"🔍 DICOM SEG 프레임 추출 시작: {seg_instance_id}")
        
        client = OrthancClient()
        
        # Orthanc에서 DICOM SEG 파일 다운로드
        seg_dicom_bytes = client.get_instance_file(seg_instance_id)
        
        # DICOM 파일 파싱
        dicom_data = io.BytesIO(seg_dicom_bytes)
        ds = pydicom.dcmread(dicom_data, force=True)
        
        # NumberOfFrames 확인
        num_frames = getattr(ds, 'NumberOfFrames', 1)
        rows = ds.Rows
        cols = ds.Columns
        
        logger.info(f"📊 DICOM SEG 정보: {num_frames} frames, {rows}×{cols}")
        
        # PixelData 추출 - pydicom 사용 (1-bit 압축 자동 처리)
        try:
            pixel_array = ds.pixel_array  # pydicom이 자동으로 언팩
            if pixel_array.ndim == 2:
                pixel_array = pixel_array[np.newaxis, ...]
            logger.info(f"   Pixel array shape: {pixel_array.shape}")
        except:
            # Fallback
            pixel_data = np.frombuffer(ds.PixelData, dtype=np.uint8)
            if ds.BitsAllocated == 1:
                pixel_array = np.unpackbits(pixel_data).reshape(num_frames, rows, cols)
            else:
                pixel_array = pixel_data.reshape(num_frames, rows, cols)
        
        # 각 프레임을 base64로 인코딩
        frames = []
        for i in range(num_frames):
            frame_data = (pixel_array[i] > 0).astype(np.uint8) * 255
            
            # PNG로 인코딩
            from PIL import Image
            img = Image.fromarray(frame_data, mode='L')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            frames.append({
                "index": i,
                "mask_base64": mask_base64
            })
        
        logger.info(f"✅ {len(frames)}개 프레임 추출 완료")
        
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

# 요청 디렉토리 (연구실 컴퓨터와 공유)
REQUEST_DIR = Path(os.getenv('INFERENCE_REQUEST_DIR', '/tmp/mri_inference_requests'))


def _create_local_inference_request(request, series_id, sequence_series_ids):
    """
    연구실 컴퓨터에서 추론 실행 요청 생성 (내부 함수)
    DRF Request 객체를 직접 처리
    """
    try:
        if len(sequence_series_ids) != 4:
            return Response({
                'success': False,
                'error': '4개 시리즈가 필요합니다.'
            }, status=400)
        
        # 요청 디렉토리 생성
        REQUEST_DIR.mkdir(exist_ok=True, parents=True)
        
        # 요청 데이터 생성
        request_data = {
            'series_ids': sequence_series_ids,
            'main_series_id': series_id,
            'requested_at': timezone.now().isoformat(),
            'status': 'pending',
            'requested_by': getattr(request.user, 'username', 'anonymous') if hasattr(request, 'user') and hasattr(request.user, 'is_authenticated') and request.user.is_authenticated else 'anonymous'
        }
        
        # 요청 파일 저장 (타임스탬프 포함하여 중복 방지)
        timestamp = int(timezone.now().timestamp() * 1000)
        request_id = f"{series_id}_{timestamp}"
        request_file = REQUEST_DIR / f"{request_id}.json"
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 요청 생성: {request_file.name}")
        logger.info(f"   - 시리즈: {sequence_series_ids}")
        logger.info(f"   - 요청자: {request_data['requested_by']}")
        
        # 워커가 처리할 때까지 대기 (최대 5분)
        import time
        max_wait_time = 300  # 5분
        check_interval = 2  # 2초마다 확인
        elapsed_time = 0
        
        logger.info("⏳ 연구실 컴퓨터 워커가 요청을 처리할 때까지 대기 중...")
        
        while elapsed_time < max_wait_time:
            time.sleep(check_interval)
            elapsed_time += check_interval
            
            # 요청 상태 확인
            try:
                with open(request_file, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
                
                current_status = current_data.get('status')
                
                if current_status == 'completed':
                    # 완료됨 - 결과 반환
                    result = current_data.get('result', {})
                    logger.info(f"✅ 추론 완료! (소요 시간: {elapsed_time}초)")
                    
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
                    # 실패
                    result = current_data.get('result', {})
                    error_msg = result.get('error', '알 수 없는 오류')
                    logger.error(f"❌ 추론 실패: {error_msg}")
                    
                    return Response({
                        'success': False,
                        'error': error_msg,
                        'request_id': request_id
                    }, status=500)
                
                elif current_status == 'processing':
                    # 처리 중
                    logger.info(f"   처리 중... ({elapsed_time}초 경과)")
                
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # 파일이 아직 생성되지 않았거나 읽기 실패
                pass
            
            # 진행률 표시 (30초마다)
            if elapsed_time % 30 == 0:
                logger.info(f"   대기 중... ({elapsed_time}/{max_wait_time}초)")
        
        # 타임아웃
        logger.warning(f"⏱️ 요청 처리 타임아웃 ({max_wait_time}초 경과)")
        return Response({
            'success': False,
            'error': f'요청 처리 시간 초과 (최대 {max_wait_time}초)',
            'request_id': request_id,
            'message': '연구실 컴퓨터 워커가 요청을 처리하지 못했습니다. 워커가 실행 중인지 확인하세요.'
        }, status=504)
        
    except Exception as e:
        logger.error(f"❌ 추론 요청 생성 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)


@api_view(['POST'])
def request_local_inference(request, series_id):
    """
    연구실 컴퓨터에서 추론 실행 요청
    
    POST /api/mri/segmentation/series/<series_id>/request-local/
    Body: {
        "sequence_series_ids": [series1_id, series2_id, series3_id, series4_id]
    }
    """
    try:
        sequence_series_ids = request.data.get("sequence_series_ids", [])
        
        if len(sequence_series_ids) != 4:
            return Response({
                'success': False,
                'error': '4개 시리즈가 필요합니다.'
            }, status=400)
        
        # 요청 디렉토리 생성
        REQUEST_DIR.mkdir(exist_ok=True, parents=True)
        
        # 요청 데이터 생성
        request_data = {
            'series_ids': sequence_series_ids,
            'main_series_id': series_id,
            'requested_at': timezone.now().isoformat(),
            'status': 'pending',
            'requested_by': getattr(request.user, 'username', 'anonymous') if hasattr(request, 'user') and hasattr(request.user, 'is_authenticated') and request.user.is_authenticated else 'anonymous'
        }
        
        # 요청 파일 저장 (타임스탬프 포함하여 중복 방지)
        timestamp = int(timezone.now().timestamp() * 1000)
        request_id = f"{series_id}_{timestamp}"
        request_file = REQUEST_DIR / f"{request_id}.json"
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 요청 생성: {request_file.name}")
        logger.info(f"   - 시리즈: {sequence_series_ids}")
        logger.info(f"   - 요청자: {request_data['requested_by']}")
        
        # 워커가 처리할 때까지 대기 (최대 5분)
        import time
        max_wait_time = 300  # 5분
        check_interval = 2  # 2초마다 확인
        elapsed_time = 0
        
        logger.info("⏳ 연구실 컴퓨터 워커가 요청을 처리할 때까지 대기 중...")
        
        while elapsed_time < max_wait_time:
            time.sleep(check_interval)
            elapsed_time += check_interval
            
            # 요청 상태 확인
            try:
                with open(request_file, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
                
                current_status = current_data.get('status')
                
                if current_status == 'completed':
                    # 완료됨 - 결과 반환
                    result = current_data.get('result', {})
                    logger.info(f"✅ 추론 완료! (소요 시간: {elapsed_time}초)")
                    
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
                    # 실패
                    result = current_data.get('result', {})
                    error_msg = result.get('error', '알 수 없는 오류')
                    logger.error(f"❌ 추론 실패: {error_msg}")
                    
                    return Response({
                        'success': False,
                        'error': error_msg,
                        'request_id': request_id
                    }, status=500)
                
                elif current_status == 'processing':
                    # 처리 중
                    logger.info(f"   처리 중... ({elapsed_time}초 경과)")
                
            except (FileNotFoundError, json.JSONDecodeError) as e:
                # 파일이 아직 생성되지 않았거나 읽기 실패
                pass
            
            # 진행률 표시 (30초마다)
            if elapsed_time % 30 == 0:
                logger.info(f"   대기 중... ({elapsed_time}/{max_wait_time}초)")
        
        # 타임아웃
        logger.warning(f"⏱️ 타임아웃: 워커가 {max_wait_time}초 내에 응답하지 않음")
        
        return Response({
            'success': False,
            'error': f'연구실 컴퓨터 워커가 {max_wait_time}초 내에 응답하지 않았습니다. 워커가 실행 중인지 확인하세요.',
            'request_id': request_id,
            'status': 'timeout',
            'note': '요청은 생성되었습니다. 나중에 /api/mri/segmentation/status/{request_id}/ 에서 상태를 확인하세요.'
        }, status=504)
        
    except Exception as e:
        logger.error(f"❌ 추론 요청 생성 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def check_inference_status(request, request_id):
    """
    추론 요청 상태 확인
    
    GET /api/mri/segmentation/status/<request_id>/
    """
    try:
        # 요청 파일 찾기
        request_files = list(REQUEST_DIR.glob(f"{request_id}.json"))
        
        if not request_files:
            return Response({
                'success': False,
                'error': '요청을 찾을 수 없습니다.',
                'request_id': request_id
            }, status=404)
        
        # 요청 데이터 읽기
        with open(request_files[0], 'r', encoding='utf-8') as f:
            request_data = json.load(f)
        
        # 상태별 메시지
        status_messages = {
            'pending': '대기 중: 연구실 컴퓨터에서 처리 대기 중입니다.',
            'processing': '처리 중: 추론이 진행 중입니다.',
            'completed': '완료: 추론이 성공적으로 완료되었습니다.',
            'failed': '실패: 추론 중 오류가 발생했습니다.'
        }
        
        current_status = request_data.get('status', 'unknown')
        
        response_data = {
            'success': True,
            'request_id': request_id,
            'status': current_status,
            'message': status_messages.get(current_status, '알 수 없는 상태'),
            'requested_at': request_data.get('requested_at'),
            'started_at': request_data.get('started_at'),
            'completed_at': request_data.get('completed_at'),
            'series_ids': request_data.get('series_ids'),
            'requested_by': request_data.get('requested_by')
        }
        
        # 결과가 있으면 포함
        if 'result' in request_data:
            result = request_data['result']
            response_data['result'] = {
                'success': result.get('success'),
                'seg_instance_id': result.get('seg_instance_id'),
                'tumor_detected': result.get('tumor_detected'),
                'tumor_volume_voxels': result.get('tumor_volume_voxels'),
                'elapsed_time_seconds': result.get('elapsed_time_seconds'),
                'error': result.get('error')
            }
        
        return Response(response_data)
        
    except Exception as e:
        logger.error(f"❌ 상태 확인 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def list_inference_requests(request):
    """
    추론 요청 목록 조회
    
    GET /api/mri/segmentation/requests/
    Query params:
        - status: pending, processing, completed, failed
        - limit: 최대 개수 (기본: 50)
    """
    try:
        # 쿼리 파라미터
        filter_status = request.GET.get('status')
        limit = int(request.GET.get('limit', 50))
        
        # 요청 파일 찾기
        request_files = sorted(REQUEST_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        requests_list = []
        for request_file in request_files[:limit]:
            try:
                with open(request_file, 'r', encoding='utf-8') as f:
                    request_data = json.load(f)
                
                # 상태 필터링
                if filter_status and request_data.get('status') != filter_status:
                    continue
                
                requests_list.append({
                    'request_id': request_file.stem,
                    'status': request_data.get('status'),
                    'requested_at': request_data.get('requested_at'),
                    'started_at': request_data.get('started_at'),
                    'completed_at': request_data.get('completed_at'),
                    'series_ids': request_data.get('series_ids'),
                    'requested_by': request_data.get('requested_by'),
                    'has_result': 'result' in request_data
                })
            except Exception as e:
                logger.warning(f"⚠️ 요청 파일 읽기 실패: {request_file.name} - {e}")
                continue
        
        return Response({
            'success': True,
            'count': len(requests_list),
            'requests': requests_list,
            'filter': {
                'status': filter_status,
                'limit': limit
            }
        })
        
    except Exception as e:
        logger.error(f"❌ 요청 목록 조회 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_pending_requests(request):
    """
    연구실 컴퓨터 워커용: 대기 중인 추론 요청 조회 (HTTP API 방식)
    
    GET /api/mri/segmentation/pending-requests/
    
    연구실 컴퓨터가 이 API를 폴링하여 요청을 가져옵니다.
    공유 디렉토리나 내부 IP가 필요 없습니다!
    """
    try:
        # 대기 중인 요청만 찾기
        request_files = sorted(REQUEST_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime)
        
        pending_requests = []
        for request_file in request_files:
            try:
                with open(request_file, 'r', encoding='utf-8') as f:
                    request_data = json.load(f)
                
                # pending 상태만 반환
                if request_data.get('status') == 'pending':
                    pending_requests.append({
                        'request_id': request_file.stem,
                        'series_ids': request_data.get('series_ids'),
                        'main_series_id': request_data.get('main_series_id'),
                        'requested_at': request_data.get('requested_at'),
                        'requested_by': request_data.get('requested_by')
                    })
            except Exception as e:
                logger.warning(f"⚠️ 요청 파일 읽기 실패: {request_file.name} - {e}")
                continue
        
        return Response({
            'success': True,
            'count': len(pending_requests),
            'requests': pending_requests
        })
        
    except Exception as e:
        logger.error(f"❌ 대기 중인 요청 조회 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_pending_inference(request):
    """
    조원님 워커 호환용: 대기 중인 추론 요청 조회 (단일 요청 반환)
    
    GET /api/inference/pending
    
    조원님의 워커가 사용하는 형식:
    - 요청이 있으면: {"id": request_id, "series_id": "...", "series_ids": [...]}
    - 요청이 없으면: {"id": null}
    """
    try:
        # 대기 중인 요청만 찾기 (가장 오래된 것부터)
        request_files = sorted(REQUEST_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime)
        
        for request_file in request_files:
            try:
                with open(request_file, 'r', encoding='utf-8') as f:
                    request_data = json.load(f)
                
                # pending 상태만 반환
                if request_data.get('status') == 'pending':
                    # 상태를 processing으로 변경
                    request_data['status'] = 'processing'
                    request_data['started_at'] = timezone.now().isoformat()
                    
                    with open(request_file, 'w', encoding='utf-8') as f:
                        json.dump(request_data, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"✅ 요청 할당: {request_file.stem}")
                    
                    # 조원님 워커 형식으로 반환
                    return Response({
                        'id': request_file.stem,
                        'series_id': request_data.get('main_series_id'),
                        'series_ids': request_data.get('series_ids', [])
                    })
            except Exception as e:
                logger.warning(f"⚠️ 요청 파일 읽기 실패: {request_file.name} - {e}")
                continue
        
        # 대기 중인 요청 없음
        return Response({'id': None})
        
    except Exception as e:
        logger.error(f"❌ 대기 중인 요청 조회 실패: {str(e)}", exc_info=True)
        return Response({
            'id': None,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def complete_inference(request, request_id):
    """
    조원님 워커 호환용: 추론 완료 결과 업로드
    
    POST /api/inference/{request_id}/complete
    Body: {
        "success": true,
        "seg_instance_id": "...",
        "tumor_detected": true,
        "tumor_volume_voxels": 1234,
        "inference_time_seconds": 45.2
    }
    """
    try:
        request_file = REQUEST_DIR / f"{request_id}.json"
        
        if not request_file.exists():
            return Response({
                'success': False,
                'error': '요청을 찾을 수 없습니다.'
            }, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            request_data = json.load(f)
        
        # 결과 업데이트
        request_data['status'] = 'completed' if request.data.get('success') else 'failed'
        request_data['completed_at'] = timezone.now().isoformat()
        request_data['result'] = {
            'success': request.data.get('success'),
            'seg_instance_id': request.data.get('seg_instance_id'),
            'tumor_detected': request.data.get('tumor_detected'),
            'tumor_volume_voxels': request.data.get('tumor_volume_voxels'),
            'elapsed_time_seconds': request.data.get('inference_time_seconds') or request.data.get('elapsed_time_seconds'),
            'error': request.data.get('error')
        }
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 완료 결과 업로드: {request_id}")
        
        return Response({
            'success': True
        })
        
    except Exception as e:
        logger.error(f"❌ 결과 업로드 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def complete_inference_request(request, request_id):
    """
    연구실 컴퓨터 워커용: 추론 완료 결과 업로드 (HTTP API 방식)
    
    POST /api/mri/segmentation/complete-request/<request_id>/
    Body: {
        "success": true,
        "seg_instance_id": "...",
        "tumor_detected": true,
        "tumor_volume_voxels": 12345,
        "elapsed_time_seconds": 30.5,
        "error": null
    }
    """
    try:
        # 요청 파일 찾기
        request_file = REQUEST_DIR / f"{request_id}.json"
        
        if not request_file.exists():
            return Response({
                'success': False,
                'error': '요청을 찾을 수 없습니다.'
            }, status=404)
        
        # 요청 데이터 읽기
        with open(request_file, 'r', encoding='utf-8') as f:
            request_data = json.load(f)
        
        # 결과 업데이트
        request_data['status'] = 'completed' if request.data.get('success') else 'failed'
        request_data['completed_at'] = timezone.now().isoformat()
        request_data['result'] = {
            'success': request.data.get('success'),
            'seg_instance_id': request.data.get('seg_instance_id'),
            'tumor_detected': request.data.get('tumor_detected'),
            'tumor_volume_voxels': request.data.get('tumor_volume_voxels'),
            'elapsed_time_seconds': request.data.get('elapsed_time_seconds'),
            'error': request.data.get('error')
        }
        
        # 파일 저장
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 추론 완료 결과 업로드: {request_id}")
        
        return Response({
            'success': True,
            'message': '결과가 성공적으로 업로드되었습니다.',
            'request_id': request_id
        })
        
    except Exception as e:
        logger.error(f"❌ 결과 업로드 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def update_request_status(request, request_id):
    """
    연구실 컴퓨터 워커용: 요청 상태 업데이트 (processing 등)
    
    POST /api/mri/segmentation/update-status/<request_id>/
    Body: {
        "status": "processing",
        "started_at": "2024-01-01T00:00:00"
    }
    """
    try:
        request_file = REQUEST_DIR / f"{request_id}.json"
        
        if not request_file.exists():
            return Response({
                'success': False,
                'error': '요청을 찾을 수 없습니다.'
            }, status=404)
        
        with open(request_file, 'r', encoding='utf-8') as f:
            request_data = json.load(f)
        
        # 상태 업데이트
        if 'status' in request.data:
            request_data['status'] = request.data['status']
        if 'started_at' in request.data:
            request_data['started_at'] = request.data['started_at']
        
        with open(request_file, 'w', encoding='utf-8') as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        
        return Response({
            'success': True,
            'request_id': request_id,
            'status': request_data.get('status')
        })
        
    except Exception as e:
        logger.error(f"❌ 상태 업데이트 실패: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
