"""
MRI 세그멘테이션 API Views (MAMA_MIA_DELIVERY_PKG 파이프라인 사용)
- Orthanc 연동: 기존 시스템 로직 유지
- 추론: 새로운 MAMA_MIA 파이프라인 사용
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import requests
import io
import logging
import os
import base64
import numpy as np
import pydicom
import tempfile
import shutil
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
        _pipeline = SegmentationInferencePipeline(
            model_path=str(MODEL_PATH),
            device="cuda" if os.getenv('USE_GPU', 'false').lower() == 'true' else "cpu",
            threshold=0.5
        )
        logger.info("Model loaded successfully!")
    return _pipeline


@api_view(['POST'])
def mri_segmentation(request, instance_id):
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
    시리즈 전체를 3D 세그멘테이션하고 Orthanc에 저장 (MAMA-MIA 파이프라인)
    
    POST /api/mri/segmentation/series/<series_id>/segment/
    Body (required): {
        "sequence_series_ids": [series1_id, series2_id, series3_id, series4_id]  // 4-channel 필수
    }
    """
    import tempfile
    import shutil
    from pathlib import Path
    import sys
    
    try:
        logger.info(f"🔍 시리즈 3D 세그멘테이션 시작 (MAMA-MIA): series_id={series_id}")
        
        client = OrthancClient()
        
        # 요청 body에서 4개 시퀀스 ID 가져오기 (필수)
        sequence_series_ids = request.data.get("sequence_series_ids", [])
        
        # 4개 시리즈 필수 체크
        if len(sequence_series_ids) != 4:
            return Response({
                "success": False,
                "error": "4개 시리즈가 모두 필요합니다. DCE-MRI 세그멘테이션을 위해서는 Seq0, Seq1, Seq2, SeqLast 시리즈가 모두 선택되어야 합니다."
            }, status=400)
        
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
