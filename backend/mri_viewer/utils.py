import os
import numpy as np
import nibabel as nib
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image


def load_nifti_file(file_path):
    """NIfTI 파일을 로드하고 numpy 배열로 반환"""
    try:
        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()
        return data, nii_img.affine, nii_img.header
    except Exception as e:
        raise Exception(f"NIfTI 파일 로드 실패: {str(e)}")


def normalize_slice(slice_data):
    """슬라이스 데이터를 0-255 범위로 정규화"""
    if slice_data.max() > slice_data.min():
        normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        return (normalized * 255).astype(np.uint8)
    return slice_data.astype(np.uint8)


def get_slice_from_volume(volume, slice_idx, axis='axial'):
    """
    3D 볼륨에서 특정 슬라이스 추출
    axis: 'axial' (z축), 'sagittal' (x축), 'coronal' (y축)
    """
    if axis == 'axial':
        slice_data = volume[:, :, slice_idx]
    elif axis == 'sagittal':
        slice_data = volume[slice_idx, :, :]
    elif axis == 'coronal':
        slice_data = volume[:, slice_idx, :]
    else:
        raise ValueError(f"Invalid axis: {axis}")
    
    return slice_data


def create_overlay(image_slice, segmentation_slice, alpha=0.5):
    """
    이미지 슬라이스와 세그멘테이션을 오버레이
    """
    # 이미지를 RGB로 변환
    image_rgb = np.stack([image_slice] * 3, axis=-1)
    
    # 세그멘테이션을 컬러맵으로 변환 (빨간색)
    overlay = image_rgb.copy()
    mask = segmentation_slice > 0
    overlay[mask, 0] = np.minimum(255, overlay[mask, 0] + 100)  # 빨간색 채널 증가
    
    # 알파 블렌딩
    result = (1 - alpha) * image_rgb + alpha * overlay
    return result.astype(np.uint8)


def numpy_to_base64(array):
    """numpy 배열을 base64 인코딩된 PNG 이미지로 변환"""
    img = Image.fromarray(array)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def get_patient_mri_data(patient_id, data_root="/Users/nogeon-u/Desktop/건양대_바이오메디컬/Django/mmm"):
    """
    환자의 MRI 데이터 경로 및 정보 반환
    """
    data_root = Path(data_root)
    
    # 이미지 파일들 찾기
    image_dir = data_root / "images" / patient_id.upper()
    if not image_dir.exists():
        raise FileNotFoundError(f"환자 데이터를 찾을 수 없습니다: {patient_id}")
    
    # NIfTI 파일들 수집
    image_files = sorted(list(image_dir.glob("*.nii.gz")))
    
    # 세그멘테이션 파일 찾기
    seg_auto = data_root / "segmentations" / "automatic" / f"{patient_id.lower()}.nii.gz"
    seg_expert = data_root / "segmentations" / "expert" / f"{patient_id.lower()}.nii.gz"
    
    # 환자 정보 파일
    patient_info_file = data_root / "patient_info_files" / f"{patient_id.lower()}.json"
    
    return {
        'image_files': [str(f) for f in image_files],
        'segmentation_auto': str(seg_auto) if seg_auto.exists() else None,
        'segmentation_expert': str(seg_expert) if seg_expert.exists() else None,
        'patient_info_file': str(patient_info_file) if patient_info_file.exists() else None,
    }


def load_mri_series(image_files):
    """
    여러 MRI 시퀀스 파일들을 로드
    """
    series_data = []
    for file_path in image_files:
        data, affine, header = load_nifti_file(file_path)
        series_data.append({
            'data': data,
            'affine': affine,
            'header': header,
            'filename': os.path.basename(file_path),
            'shape': data.shape
        })
    return series_data

