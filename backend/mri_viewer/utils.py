import os
import numpy as np
import nibabel as nib
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
from datetime import datetime
import uuid


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


def nifti_to_dicom_slices(nifti_file, patient_id=None, patient_name=None, image_type=None):
    """
    NIfTI 파일을 DICOM 슬라이스들로 변환
    
    Args:
        nifti_file: 파일 경로(str/Path) 또는 파일 객체(BytesIO 등)
        patient_id: 환자 ID (선택사항)
        patient_name: 환자 이름 (선택사항)
        image_type: 영상 유형 ('유방촬영술 영상', '병리 영상', 'MRI 영상') - Study/Series 구분용
    
    Returns:
        List[bytes]: DICOM 인스턴스들의 바이트 데이터 리스트
    """
    # NIfTI 파일 로드
    import tempfile
    import os
    
    if hasattr(nifti_file, 'read'):
        # 파일 객체인 경우 (BytesIO 등)
        # nibabel.load()는 파일 경로만 받으므로 임시 파일이 필요
        if hasattr(nifti_file, 'seek'):
            nifti_file.seek(0)
        
        # BytesIO 내용을 읽기
        file_data = nifti_file.read()
        if len(file_data) == 0:
            raise ValueError("NIfTI file data is empty")
        
        if hasattr(nifti_file, 'seek'):
            nifti_file.seek(0)  # 다시 처음으로
        
        # 파일 확장자 확인
        file_suffix = '.nii.gz'
        if hasattr(nifti_file, 'name'):
            if nifti_file.name.endswith('.nii.gz'):
                file_suffix = '.nii.gz'
            elif nifti_file.name.endswith('.nii'):
                file_suffix = '.nii'
        
        # 임시 파일 생성 (여러 방법 시도)
        tmp_file_path = None
        temp_dir = None
        
        # 방법 1: 시스템 임시 디렉토리 사용
        try:
            temp_dir = tempfile.gettempdir()
            if not os.path.exists(temp_dir):
                raise OSError(f"System temp directory does not exist: {temp_dir}")
            if not os.access(temp_dir, os.W_OK):
                raise OSError(f"No write permission to temp directory: {temp_dir}")
        except Exception as e:
            # 방법 2: 현재 작업 디렉토리의 temp_nifti 폴더 사용
            try:
                temp_dir = os.path.join(os.getcwd(), 'temp_nifti')
                os.makedirs(temp_dir, exist_ok=True)
                if not os.access(temp_dir, os.W_OK):
                    raise OSError(f"No write permission to temp directory: {temp_dir}")
            except Exception as e2:
                # 방법 3: 프로젝트 루트의 temp_nifti 폴더 사용
                try:
                    # Django 프로젝트 루트 찾기 (settings.py가 있는 디렉토리)
                    try:
                        from django.conf import settings
                        if hasattr(settings, 'BASE_DIR'):
                            temp_dir = os.path.join(settings.BASE_DIR, 'temp_nifti')
                        else:
                            raise AttributeError("BASE_DIR not found")
                    except (ImportError, AttributeError):
                        # Django가 초기화되지 않았거나 BASE_DIR이 없으면 현재 파일의 상위 디렉토리 사용
                        current_file_dir = os.path.dirname(os.path.abspath(__file__))
                        project_root = os.path.dirname(os.path.dirname(current_file_dir))
                        temp_dir = os.path.join(project_root, 'temp_nifti')
                    
                    os.makedirs(temp_dir, exist_ok=True)
                    if not os.access(temp_dir, os.W_OK):
                        raise OSError(f"No write permission to temp directory: {temp_dir}")
                except Exception as e3:
                    raise OSError(f"Could not create or access temp directory. Tried: {tempfile.gettempdir()}, {os.path.join(os.getcwd(), 'temp_nifti')}, {temp_dir}. Errors: {e}, {e2}, {e3}")
        
        # 임시 파일 경로 생성
        tmp_file_path = os.path.join(temp_dir, f"nifti_{uuid.uuid4().hex}{file_suffix}")
        
        try:
            # 파일 쓰기 (명시적으로 바이너리 모드)
            with open(tmp_file_path, 'wb') as tmp_file:
                tmp_file.write(file_data)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # 디스크에 강제 쓰기
            
            # 파일이 실제로 존재하고 읽을 수 있는지 확인
            if not os.path.exists(tmp_file_path):
                raise IOError(f"Temporary file was not created: {tmp_file_path}")
            if not os.access(tmp_file_path, os.R_OK):
                raise IOError(f"Temporary file is not readable: {tmp_file_path}")
            
            # 파일 크기 확인
            file_size = os.path.getsize(tmp_file_path)
            if file_size == 0:
                raise IOError(f"Temporary file is empty: {tmp_file_path}")
            if file_size != len(file_data):
                raise IOError(f"Temporary file size mismatch: expected {len(file_data)}, got {file_size}")
            
            # 임시 파일에서 NIfTI 로드 (파일이 확실히 존재하는지 다시 확인)
            if not os.path.exists(tmp_file_path):
                raise IOError(f"Temporary file disappeared before loading: {tmp_file_path}")
            
            # nibabel.load() 호출
            nii_img = nib.load(tmp_file_path)
            
            # 로드 성공 후 파일 삭제 (finally 블록에서도 삭제 시도하지만 여기서도 삭제)
            # 파일이 성공적으로 로드되었으므로 삭제 가능
            try:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
            except:
                pass  # 삭제 실패는 무시 (finally에서 다시 시도)
            
        except Exception as load_error:
            # 오류 발생 시 상세 정보 포함
            error_msg = f"Failed to load NIfTI from temporary file: {load_error}"
            if tmp_file_path:
                error_msg += f"\nTemp file path: {tmp_file_path}"
                error_msg += f"\nTemp file exists: {os.path.exists(tmp_file_path) if tmp_file_path else False}"
                if tmp_file_path and os.path.exists(tmp_file_path):
                    error_msg += f"\nTemp file size: {os.path.getsize(tmp_file_path)}"
                error_msg += f"\nTemp directory: {temp_dir}"
                error_msg += f"\nTemp directory exists: {os.path.exists(temp_dir) if temp_dir else False}"
                error_msg += f"\nTemp directory writable: {os.access(temp_dir, os.W_OK) if temp_dir and os.path.exists(temp_dir) else False}"
            raise IOError(error_msg) from load_error
            
        finally:
            # 임시 파일 삭제 (이미 삭제되었을 수 있음)
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as cleanup_error:
                    # 삭제 실패는 경고만 출력 (치명적이지 않음)
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Could not delete temporary file {tmp_file_path}: {cleanup_error}")
    elif isinstance(nifti_file, (str, Path)):
        # 파일 경로인 경우
        nii_img = nib.load(str(nifti_file))
    else:
        raise ValueError(f"Unsupported nifti_file type: {type(nifti_file)}. Expected file path (str/Path) or file-like object (BytesIO)")
    
    volume = nii_img.get_fdata()
    header = nii_img.header
    
    # 환자 정보 설정
    if patient_id is None:
        patient_id = "UNKNOWN"
    if patient_name is None:
        patient_name = patient_id
    
    # 영상 유형에 따른 Study/Series 설정
    image_type_map = {
        '유방촬영술 영상': {
            'study_description': '유방촬영술',
            'series_description': 'Mammography Series',
            'modality': 'MG',
            'sop_class_uid': '1.2.840.10008.5.1.4.1.1.1.2'  # Digital Mammography X-Ray Image Storage
        },
        '병리 영상': {
            'study_description': '병리 영상',
            'series_description': 'Pathology Series',
            'modality': 'SM',  # Slide Microscopy (병리 슬라이드) 또는 'OT' (Other)
            'sop_class_uid': '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage (기본값)
        },
        'MRI 영상': {
            'study_description': 'MRI Study',
            'series_description': 'MRI Series',
            'modality': 'MR',
            'sop_class_uid': '1.2.840.10008.5.1.4.1.1.4'  # MR Image Storage
        }
    }
    
    # 영상 유형별 설정 (기본값: MRI)
    settings = image_type_map.get(image_type, image_type_map['MRI 영상'])
    
    # DICOM 메타데이터 생성
    # 같은 영상 유형은 같은 StudyInstanceUID를 사용하도록 (환자 ID + 영상 유형 기반)
    # 실제로는 매번 새 Study를 생성하지만, StudyDescription으로 구분 가능
    study_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    
    # 볼륨의 shape 확인 및 처리
    if len(volume.shape) == 2:
        # 2D 이미지인 경우
        volume = volume[:, :, np.newaxis]  # 3D로 변환
        num_slices = 1
    elif len(volume.shape) == 3:
        num_slices = volume.shape[2]
    elif len(volume.shape) == 4:
        num_slices = volume.shape[2]
        volume = volume[:, :, :, 0]  # 첫 번째 시간 단계만 사용
    else:
        raise ValueError(f"Unsupported volume shape: {volume.shape}")
    
    dicom_slices = []
    
    for slice_idx in range(num_slices):
        # 슬라이스 추출
        slice_data = volume[:, :, slice_idx]
        
        # 픽셀 값을 정수형으로 변환 (DICOM은 정수형 필요)
        # NIfTI 데이터를 Hounsfield Unit 범위로 가정
        if slice_data.dtype != np.uint16:
            # 데이터를 적절한 범위로 스케일링
            min_val = slice_data.min()
            max_val = slice_data.max()
            
            if max_val > 32767:
                # float 데이터인 경우 -1024 to 3071 (일반적인 CT 범위)로 매핑
                slice_data = np.clip(slice_data, -1024, 3071)
                slice_data = slice_data.astype(np.int16)
            else:
                slice_data = slice_data.astype(np.int16)
        
        # DICOM 데이터셋 생성
        ds = Dataset()
        
        # 필수 DICOM 태그 (DICOM 태그 형식으로 명시적 설정)
        from pydicom.tag import Tag
        ds.PatientID = str(patient_id)  # (0010,0020)
        ds.PatientName = str(patient_name)  # (0010,0010)
        ds.PatientBirthDate = ""  # (0010,0030)
        ds.PatientSex = ""  # (0010,0040)
        
        # 디버깅: PatientID 확인
        print(f"DICOM Slice {slice_idx + 1}: PatientID={ds.PatientID}, PatientName={ds.PatientName}")
        
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate = datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.now().strftime("%H%M%S")
        ds.StudyID = str(uuid.uuid4())[:8]
        ds.StudyDescription = settings['study_description']
        
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber = "1"
        ds.SeriesDescription = settings['series_description']
        ds.Modality = settings['modality']
        
        ds.InstanceNumber = str(slice_idx + 1)
        ds.SOPInstanceUID = generate_uid()
        ds.SOPClassUID = settings['sop_class_uid']
        
        # 이미지 파라미터
        ds.Rows = slice_data.shape[0]
        ds.Columns = slice_data.shape[1]
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1 if slice_data.dtype == np.int16 else 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        
        # 슬라이스 위치 (간단한 추정)
        try:
            if hasattr(header, 'get'):
                pixdim = header.get('pixdim', [1, 1, 1, 1])
                slice_thickness = float(pixdim[3]) if len(pixdim) > 3 else 1.0
            else:
                slice_thickness = 1.0
        except:
            slice_thickness = 1.0
        
        slice_location = slice_idx * slice_thickness
        ds.SliceLocation = str(slice_location)
        ds.SliceThickness = str(slice_thickness)
        
        # 픽셀 데이터 (numpy 배열을 직접 할당)
        ds.PixelData = slice_data.tobytes()
        
        # 파일로 저장 (메모리)
        buffer = BytesIO()
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9"
        
        ds.file_meta = file_meta
        ds.is_implicit_VR = False
        ds.is_little_endian = True
        
        pydicom.dcmwrite(buffer, ds, write_like_original=False)
        dicom_slices.append(buffer.getvalue())
    
    return dicom_slices

