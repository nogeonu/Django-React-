"""
MRI 세그멘테이션 API Views (새로운 파이프라인 사용)
조원 코드 기반으로 재구성
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
from pathlib import Path
from .orthanc_client import OrthancClient
import sys

# 새로운 세그멘테이션 모듈 import
sys.path.insert(0, str(Path(__file__).parent.parent / "mri_segmentation_new"))
from dicom_nifti_converter import dicom_series_to_nifti, nifti_to_dicom_seg
from inference_pipeline import SegmentationInferencePipeline

logger = logging.getLogger(__name__)

# Orthanc 설정
ORTHANC_URL = os.getenv('ORTHANC_URL', 'http://34.42.223.43:8042')
ORTHANC_USER = os.getenv('ORTHANC_USER', 'admin')
ORTHANC_PASSWORD = os.getenv('ORTHANC_PASSWORD', 'admin123')

# 모델 경로
MODEL_PATH = Path(__file__).parent.parent / "mri_segmentation_new" / "checkpoints" / "best_model.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = Path(__file__).parent.parent / "mri_segmentation_new" / "best_model.pth"

# 전역 추론 파이프라인 (한 번만 로드)
_pipeline = None

def get_pipeline():
    """추론 파이프라인 싱글톤"""
    global _pipeline
    if _pipeline is None:
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
    시리즈 전체를 3D 세그멘테이션하고 Orthanc에 저장 (4-channel, 새로운 파이프라인)
    
    POST /api/mri/segmentation/series/<series_id>/segment/
    Body (required): {
        "sequence_series_ids": [series1_id, series2_id, series3_id, series4_id]  // 4-channel 필수
    }
    """
    try:
        logger.info(f"🔍 시리즈 3D 세그멘테이션 시작 (새 파이프라인): series_id={series_id}")
        
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
        
        # 각 시퀀스의 모든 슬라이스 DICOM 파일 다운로드
        logger.info("📥 Orthanc에서 4개 시퀀스의 DICOM 파일 다운로드 중...")
        dicom_sequences = []  # [[seq1_slice1, seq1_slice2, ...], [seq2_slice1, ...], ...]
        
        for seq_idx, seq_series_id in enumerate(sequence_series_ids):
            seq_info = client.get(f"/series/{seq_series_id}")
            seq_instances = seq_info.get("Instances", [])
            
            if len(seq_instances) == 0:
                return Response({
                    "success": False,
                    "error": f"시퀀스 {seq_idx+1}에 슬라이스가 없습니다."
                }, status=400)
            
            # 각 인스턴스의 DICOM 파일 다운로드
            seq_dicom_files = []
            for instance_id in seq_instances:
                dicom_bytes = client.get_instance_file(instance_id)
                seq_dicom_files.append(dicom_bytes)
            
            dicom_sequences.append(seq_dicom_files)
            logger.info(f"✅ 시퀀스 {seq_idx+1}/4: {len(seq_dicom_files)}개 슬라이스 다운로드 완료")
        
        # 1. DICOM → NIfTI 변환
        logger.info("🔄 DICOM → NIfTI 변환 중...")
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_nifti:
            nifti_path = tmp_nifti.name
        
        try:
            nifti_path, metadata = dicom_series_to_nifti(
                dicom_sequences=dicom_sequences,
                output_path=nifti_path
            )
            logger.info(f"✅ NIfTI 변환 완료: {nifti_path}, Shape: {metadata['shape']}")
        except Exception as e:
            logger.error(f"❌ DICOM → NIfTI 변환 실패: {e}", exc_info=True)
            return Response({
                "success": False,
                "error": f"DICOM → NIfTI 변환 실패: {str(e)}"
            }, status=500)
        
        # 2. 세그멘테이션 추론
        logger.info("🧠 세그멘테이션 추론 시작...")
        pipeline = get_pipeline()
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_seg:
            seg_nifti_path = tmp_seg.name
        
        try:
            result = pipeline.predict(
                image_path=nifti_path,
                output_path=seg_nifti_path
            )
            logger.info(f"✅ 세그멘테이션 완료: Tumor detected={result['tumor_detected']}, Volume={result['tumor_volume_voxels']} voxels")
        except Exception as e:
            logger.error(f"❌ 세그멘테이션 추론 실패: {e}", exc_info=True)
            return Response({
                "success": False,
                "error": f"세그멘테이션 추론 실패: {str(e)}"
            }, status=500)
        finally:
            # 임시 NIfTI 파일 정리
            try:
                os.unlink(nifti_path)
            except:
                pass
        
        # 3. NIfTI → DICOM SEG 변환
        logger.info("🔄 NIfTI → DICOM SEG 변환 중...")
        
        # 참조 DICOM 파일들을 임시 파일로 저장
        reference_dicom_paths = []
        try:
            for slice_bytes in dicom_sequences[0]:  # 첫 번째 시퀀스 사용
                tmp_dicom = tempfile.NamedTemporaryFile(suffix='.dcm', delete=False)
                tmp_dicom.write(slice_bytes)
                tmp_dicom.close()
                reference_dicom_paths.append(tmp_dicom.name)
            
            with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp_seg_dicom:
                seg_dicom_path = tmp_seg_dicom.name
            
            try:
                seg_dicom_path = nifti_to_dicom_seg(
                    nifti_mask_path=seg_nifti_path,
                    reference_dicom_paths=reference_dicom_paths,
                    output_path=seg_dicom_path
                )
                logger.info(f"✅ DICOM SEG 변환 완료: {seg_dicom_path}")
            except Exception as e:
                logger.error(f"❌ NIfTI → DICOM SEG 변환 실패: {e}", exc_info=True)
                # pydicom-seg가 없을 수 있으므로 fallback 사용
                logger.warning("pydicom-seg 변환 실패, utils.py의 nifti_to_dicom_slices 사용")
                from .utils import nifti_to_dicom_slices
                
                # 첫 번째 시퀀스의 첫 번째 DICOM에서 환자 정보 추출
                first_dicom = pydicom.dcmread(io.BytesIO(dicom_sequences[0][0]))
                patient_id = str(first_dicom.get('PatientID', ''))
                patient_name = str(first_dicom.get('PatientName', ''))
                
                # NIfTI 파일 경로를 직접 사용 (BytesIO 대신)
                dicom_slices = nifti_to_dicom_slices(
                    seg_nifti_path,  # 파일 경로 직접 전달
                    patient_id=patient_id,
                    patient_name=patient_name,
                    image_type='MRI 영상',
                    orthanc_client=client
                )
                
                # DICOM 슬라이스를 Orthanc에 업로드
                seg_instance_id = None
                for dicom_slice in dicom_slices:
                    upload_result = client.upload_dicom(dicom_slice)
                    if seg_instance_id is None:
                        seg_instance_id = upload_result.get('ID')
                
                logger.info(f"✅ DICOM 슬라이스로 변환 및 업로드 완료: {seg_instance_id}")
                
                return Response({
                    'success': True,
                    'series_id': series_id,
                    'total_slices': len(dicom_sequences[0]),
                    'tumor_detected': result['tumor_detected'],
                    'tumor_volume_voxels': result['tumor_volume_voxels'],
                    'seg_instance_id': seg_instance_id,
                    'saved_to_orthanc': True
                })
        finally:
            # 임시 파일 정리
            try:
                os.unlink(seg_nifti_path)
                for path in reference_dicom_paths:
                    os.unlink(path)
            except:
                pass
        
        # 4. DICOM SEG를 Orthanc에 업로드
        logger.info("📤 DICOM SEG를 Orthanc에 업로드 중...")
        try:
            with open(seg_dicom_path, 'rb') as f:
                seg_dicom_bytes = f.read()
            
            upload_result = client.upload_dicom(seg_dicom_bytes)
            seg_instance_id = upload_result.get('ID')
            
            logger.info(f"✅ Orthanc 업로드 완료: {seg_instance_id}")
        except Exception as e:
            logger.error(f"❌ Orthanc 업로드 실패: {e}", exc_info=True)
            return Response({
                "success": False,
                "error": f"Orthanc 업로드 실패: {str(e)}"
            }, status=500)
        finally:
            # 임시 DICOM SEG 파일 정리
            try:
                os.unlink(seg_dicom_path)
            except:
                pass
        
        return Response({
            'success': True,
            'series_id': series_id,
            'total_slices': len(dicom_sequences[0]),
            'tumor_detected': result['tumor_detected'],
            'tumor_volume_voxels': result['tumor_volume_voxels'],
            'seg_instance_id': seg_instance_id,
            'saved_to_orthanc': True
        })
        
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
        
        # PixelData 추출
        if not hasattr(ds, 'PixelData'):
            raise Exception("PixelData가 없습니다")
        
        pixel_array = np.frombuffer(ds.PixelData, dtype=np.uint8)
        frame_size = rows * cols
        
        # 각 프레임을 base64로 인코딩
        frames = []
        for i in range(num_frames):
            start_idx = i * frame_size
            end_idx = start_idx + frame_size
            frame_data = pixel_array[start_idx:end_idx].reshape(rows, cols)
            
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
