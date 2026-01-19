"""
DICOM ↔ NIfTI 변환 유틸리티
Orthanc DICOM 파일을 NIfTI로 변환하고, 세그멘테이션 결과를 DICOM SEG로 변환
"""
import numpy as np
import pydicom
import nibabel as nib
from pathlib import Path
import tempfile
import os
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def dicom_series_to_nifti(
    dicom_sequences: List[List[bytes]],
    output_path: Optional[str] = None
) -> Tuple[str, dict]:
    """
    DICOM 시리즈를 NIfTI 파일로 변환 (4채널 DCE-MRI)
    
    Args:
        dicom_sequences: DICOM 파일 바이트 리스트의 리스트
                        [[seq1_slice1, seq1_slice2, ...], [seq2_slice1, ...], ...]
                        각 내부 리스트는 하나의 시퀀스의 모든 슬라이스
        output_path: 출력 NIfTI 파일 경로 (None이면 임시 파일 생성)
    
    Returns:
        (nifti_path, metadata): NIfTI 파일 경로와 메타데이터 딕셔너리
    """
    if len(dicom_sequences) != 4:
        raise ValueError(f"4개 시퀀스가 필요합니다. 현재 {len(dicom_sequences)}개 제공됨")
    
    # 각 시퀀스의 3D 볼륨 구성
    volumes = []
    first_ds = None
    
    for seq_idx, sequence_slices in enumerate(dicom_sequences):
        if len(sequence_slices) == 0:
            raise ValueError(f"시퀀스 {seq_idx+1}에 슬라이스가 없습니다")
        
        # 각 슬라이스의 DICOM 파일 로드
        seq_dicoms = []
        for slice_bytes in sequence_slices:
            from io import BytesIO
            dicom_io = BytesIO(slice_bytes)
            ds = pydicom.dcmread(dicom_io)
            seq_dicoms.append(ds)
        
        # InstanceNumber 또는 SliceLocation으로 정렬
        seq_dicoms.sort(key=lambda x: (
            float(x.get('SliceLocation', 0)) if 'SliceLocation' in x else 0,
            int(x.get('InstanceNumber', 0))
        ))
        
        # 첫 번째 DICOM 저장 (메타데이터용)
        if first_ds is None:
            first_ds = seq_dicoms[0]
        
        # 픽셀 배열 추출 및 3D 볼륨 구성
        pixel_arrays = []
        for ds in seq_dicoms:
            pixel_array = ds.pixel_array.astype(np.float32)
            pixel_arrays.append(pixel_array)
        
        volume = np.stack(pixel_arrays, axis=0)  # [Z, H, W]
        volumes.append(volume)
        
        logger.info(f"시퀀스 {seq_idx+1}/4: {len(seq_dicoms)}개 슬라이스, 볼륨 shape: {volume.shape}")
    
    # 4채널 볼륨 결합: [4, Z, H, W]
    # 모든 시퀀스가 같은 Z 차원을 가져야 함
    z_dims = [vol.shape[0] for vol in volumes]
    if len(set(z_dims)) > 1:
        logger.warning(f"시퀀스들의 Z 차원이 다릅니다: {z_dims}. 최소값으로 맞춥니다.")
        min_z = min(z_dims)
        volumes = [vol[:min_z] for vol in volumes]
    
    multi_channel_volume = np.stack(volumes, axis=0)  # [4, Z, H, W]
    
    # Affine 행렬 생성 (DICOM 좌표계 → RAS 좌표계)
    spacing = [
        float(first_ds.get('PixelSpacing', [1.0, 1.0])[0]),
        float(first_ds.get('PixelSpacing', [1.0, 1.0])[1]),
        float(first_ds.get('SpacingBetweenSlices', first_ds.get('SliceThickness', 1.0)))
    ]
    
    # DICOM 좌표계에서 RAS 좌표계로 변환
    affine = np.eye(4)
    affine[0, 0] = -spacing[0]  # X 방향 반전 (DICOM → RAS)
    affine[1, 1] = -spacing[1]  # Y 방향 반전
    affine[2, 2] = spacing[2]
    
    # Image Position Patient에서 원점 설정
    if 'ImagePositionPatient' in first_ds:
        pos = first_ds.ImagePositionPatient
        affine[0, 3] = -float(pos[0])  # X 반전
        affine[1, 3] = -float(pos[1])  # Y 반전
        affine[2, 3] = float(pos[2])
    
    # NIfTI 파일로 저장
    if output_path is None:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
        output_path = tmp_file.name
        tmp_file.close()
    
    nifti_img = nib.Nifti1Image(multi_channel_volume, affine)
    nib.save(nifti_img, output_path)
    
    # 메타데이터 저장
    metadata = {
        'spacing': spacing,
        'affine': affine,
        'shape': multi_channel_volume.shape,
        'patient_id': str(first_ds.get('PatientID', '')),
        'patient_name': str(first_ds.get('PatientName', '')),
        'study_instance_uid': str(first_ds.get('StudyInstanceUID', '')),
        'series_instance_uid': str(first_ds.get('SeriesInstanceUID', ''))
    }
    
    logger.info(f"NIfTI 파일 생성 완료: {output_path}, Shape: {multi_channel_volume.shape}")
    
    return output_path, metadata


def nifti_to_dicom_seg(
    nifti_mask_path: str,
    reference_dicom_paths: List[str],
    output_path: Optional[str] = None
) -> str:
    """
    NIfTI 세그멘테이션 마스크를 DICOM SEG로 변환
    
    Args:
        nifti_mask_path: NIfTI 세그멘테이션 마스크 파일 경로
        reference_dicom_paths: 참조 DICOM 파일 경로 리스트 (첫 번째 시퀀스)
        output_path: 출력 DICOM SEG 파일 경로
    
    Returns:
        output_path: 생성된 DICOM SEG 파일 경로
    """
    try:
        from pydicom.uid import generate_uid
        from pydicom_seg import write_segmentation
    except ImportError:
        try:
            # 대체 import 시도
            from pydicomseg import write_segmentation
        except ImportError:
            raise ImportError("pydicom-seg 모듈이 설치되지 않았습니다. pip install pydicom-seg==0.4.1")
    
    # NIfTI 마스크 로드
    nifti_img = nib.load(nifti_mask_path)
    mask_data = nifti_img.get_fdata()  # [Z, H, W] 또는 [1, Z, H, W]
    
    if len(mask_data.shape) == 4:
        mask_data = mask_data[0]  # 첫 번째 채널만 사용
    
    # 참조 DICOM 로드
    reference_dicoms = [pydicom.dcmread(path) for path in reference_dicom_paths]
    reference_dicoms.sort(key=lambda x: int(x.get('InstanceNumber', 0)))
    
    # DICOM SEG 생성
    if output_path is None:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.dcm', delete=False)
        output_path = tmp_file.name
        tmp_file.close()
    
    # pydicom-seg를 사용하여 SEG 파일 생성
    # Multi-frame SEG 형식으로 생성
    segmentation = write_segmentation(
        mask_data.astype(np.uint8),
        reference_dicoms,
        output_path=output_path,
        segment_label="Tumor",
        segment_description="Tumor segmentation from SwinUNETR",
        algorithm_name="SwinUNETR+LoRA",
        algorithm_version="1.0"
    )
    
    logger.info(f"DICOM SEG 파일 생성 완료: {output_path}")
    
    return output_path
