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
import logging

logger = logging.getLogger(__name__)


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


def nifti_to_dicom_slices(nifti_file, patient_id=None, patient_name=None, image_type=None, orthanc_client=None):
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
    import logging
    
    logger = logging.getLogger(__name__)
    
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
        
        logger.info(f"Processing NIfTI file object, data size: {len(file_data)} bytes")
        
        # 방법 1: 시스템 임시 디렉토리 사용
        try:
            temp_dir = tempfile.gettempdir()
            logger.info(f"Trying system temp directory: {temp_dir}")
            if not os.path.exists(temp_dir):
                raise OSError(f"System temp directory does not exist: {temp_dir}")
            if not os.access(temp_dir, os.W_OK):
                raise OSError(f"No write permission to temp directory: {temp_dir}")
            logger.info(f"Using system temp directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"System temp directory failed: {e}")
            # 방법 2: 현재 작업 디렉토리의 temp_nifti 폴더 사용
            try:
                temp_dir = os.path.join(os.getcwd(), 'temp_nifti')
                logger.info(f"Trying current directory temp: {temp_dir}")
                os.makedirs(temp_dir, exist_ok=True)
                if not os.access(temp_dir, os.W_OK):
                    raise OSError(f"No write permission to temp directory: {temp_dir}")
                logger.info(f"Using current directory temp: {temp_dir}")
            except Exception as e2:
                logger.warning(f"Current directory temp failed: {e2}")
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
                    
                    logger.info(f"Trying project root temp: {temp_dir}")
                    os.makedirs(temp_dir, exist_ok=True)
                    if not os.access(temp_dir, os.W_OK):
                        raise OSError(f"No write permission to temp directory: {temp_dir}")
                    logger.info(f"Using project root temp: {temp_dir}")
                except Exception as e3:
                    logger.error(f"All temp directory attempts failed. Errors: {e}, {e2}, {e3}")
                    raise OSError(f"Could not create or access temp directory. Tried: {tempfile.gettempdir()}, {os.path.join(os.getcwd(), 'temp_nifti')}, {temp_dir}. Errors: {e}, {e2}, {e3}")
        
        # 임시 파일 경로 생성
        tmp_file_path = os.path.join(temp_dir, f"nifti_{uuid.uuid4().hex}{file_suffix}")
        logger.info(f"Creating temporary file: {tmp_file_path}")
        
        try:
            # 파일 쓰기 (명시적으로 바이너리 모드)
            logger.info(f"Writing {len(file_data)} bytes to temporary file")
            with open(tmp_file_path, 'wb') as tmp_file:
                tmp_file.write(file_data)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # 디스크에 강제 쓰기
            logger.info(f"Temporary file written successfully")
            
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
            
            logger.info(f"Loading NIfTI from: {tmp_file_path}")
            # nibabel.load() 호출
            nii_img = nib.load(tmp_file_path)
            logger.info(f"NIfTI loaded successfully, shape: {nii_img.shape if hasattr(nii_img, 'shape') else 'N/A'}")
            
            # 중요: get_fdata()는 지연 로딩이므로 파일이 필요함
            # 데이터를 먼저 메모리에 로드한 후에 파일 삭제
            logger.info("Loading NIfTI data into memory (get_fdata)...")
            volume_data = nii_img.get_fdata()  # 데이터를 메모리로 로드
            logger.info(f"Data loaded into memory, shape: {volume_data.shape}")
            
            # 헤더와 affine도 미리 읽기
            header_data = nii_img.header
            affine_data = nii_img.affine
            
            # 데이터와 헤더를 메모리에 로드했으므로 이제 파일 삭제 가능
            # 하지만 안전을 위해 함수 종료 전까지 유지
            # (finally 블록에서 삭제)
            
            # 메모리에 로드된 데이터를 변수에 저장 (try 블록 밖에서 사용)
            volume = volume_data
            header = header_data
            affine = affine_data
            
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
            # 임시 파일 삭제 (데이터가 메모리에 로드된 후에만 삭제)
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                    logger.info(f"Temporary file deleted: {tmp_file_path}")
                except Exception as cleanup_error:
                    # 삭제 실패는 경고만 출력 (치명적이지 않음)
                    logger.warning(f"Could not delete temporary file {tmp_file_path}: {cleanup_error}")
        
        # volume과 header는 이미 try 블록 안에서 메모리에 로드되어 변수에 저장됨
        
    elif isinstance(nifti_file, (str, Path)):
        # 파일 경로인 경우 (파일이 계속 존재하므로 지연 로딩 가능)
        nii_img = nib.load(str(nifti_file))
        volume = nii_img.get_fdata()
        header = nii_img.header
        affine = nii_img.affine
    else:
        raise ValueError(f"Unsupported nifti_file type: {type(nifti_file)}. Expected file path (str/Path) or file-like object (BytesIO)")
    
    # volume과 header는 이미 위에서 설정됨 (BytesIO인 경우 try 블록에서, 파일 경로인 경우 elif에서)
    
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
    # 같은 환자는 하나의 Study로 통합 (기존 StudyInstanceUID 재사용)
    # orthanc_client가 제공되면 기존 Study 찾기 시도
    study_instance_uid = None
    if orthanc_client is not None and patient_id:
        try:
            existing_uid = orthanc_client.get_existing_study_instance_uid(patient_id)
            if existing_uid:
                study_instance_uid = existing_uid
                logger.info(f"Reusing existing StudyInstanceUID for patient {patient_id}: {existing_uid[:20]}...")
        except Exception as e:
            logger.warning(f"Failed to get existing StudyInstanceUID, creating new one: {e}")
    
    # 기존 Study가 없으면 새로 생성
    if study_instance_uid is None:
        study_instance_uid = generate_uid()
        logger.info(f"Creating new StudyInstanceUID for patient {patient_id}: {study_instance_uid[:20]}...")
    
    # Series는 항상 새로 생성 (같은 Modality라도 업로드 시점이 다르면 다른 Series)
    series_instance_uid = generate_uid()
    
    # FrameOfReferenceUID 생성 (Study당 하나)
    frame_of_reference_uid = generate_uid()
    
    # SeriesNumber 계산 (기존 Series 개수 확인)
    series_number = 1
    if orthanc_client is not None and study_instance_uid:
        try:
            series_number = orthanc_client.get_next_series_number(study_instance_uid)
            logger.info(f"Using SeriesNumber {series_number} for new series")
        except Exception as e:
            logger.warning(f"Failed to get next series number, using 1: {e}")
            series_number = 1
    
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
    
    # NIfTI 헤더의 추가 정보 추출 (descrip, cal_min, cal_max, intent_code 등) - 한 번만 추출
    nifti_metadata = {}
    if hasattr(header, 'get'):
        try:
            # descrip: 설명 텍스트 (80자 제한)
            descrip = header.get('descrip', b'')
            if isinstance(descrip, bytes):
                descrip = descrip.decode('utf-8', errors='ignore').strip('\x00').strip()
            elif isinstance(descrip, np.ndarray):
                descrip = descrip.tobytes().decode('utf-8', errors='ignore').strip('\x00').strip()
            else:
                descrip = str(descrip).strip()
            if descrip:
                nifti_metadata['descrip'] = descrip[:80]  # 80자 제한
        except Exception as e:
            logger.debug(f"Failed to extract NIfTI descrip: {e}")
        
        try:
            # cal_min, cal_max: 캘리브레이션 값
            cal_min = header.get('cal_min', 0)
            cal_max = header.get('cal_max', 0)
            if cal_min != 0 or cal_max != 0:
                nifti_metadata['cal_min'] = float(cal_min)
                nifti_metadata['cal_max'] = float(cal_max)
        except Exception as e:
            logger.debug(f"Failed to extract NIfTI cal_min/cal_max: {e}")
        
        try:
            # intent_code: 데이터 의도 코드
            intent_code = header.get('intent_code', 0)
            if intent_code != 0:
                nifti_metadata['intent_code'] = int(intent_code)
        except Exception as e:
            logger.debug(f"Failed to extract NIfTI intent_code: {e}")
    
    # NIfTI 메타데이터 추출 완료 로그
    if nifti_metadata:
        logger.info(f"📋 NIfTI 헤더 추가 정보 추출:")
        if nifti_metadata.get('descrip'):
            logger.info(f"  - descrip: {nifti_metadata['descrip'][:50]}...")
        if nifti_metadata.get('cal_min') is not None:
            logger.info(f"  - cal_min: {nifti_metadata['cal_min']}, cal_max: {nifti_metadata.get('cal_max')}")
        if nifti_metadata.get('intent_code'):
            logger.info(f"  - intent_code: {nifti_metadata['intent_code']}")
    
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
        
        # 한글 지원을 위해 문자셋 설정 (UTF-8)
        ds.SpecificCharacterSet = 'ISO_IR 192'  # UTF-8
        
        # 필수 DICOM 태그 (DICOM 태그 형식으로 명시적 설정)
        from pydicom.tag import Tag
        ds.PatientID = str(patient_id)  # (0010,0020)
        ds.PatientName = str(patient_name)  # (0010,0010)
        ds.PatientBirthDate = ""  # (0010,0030)
        ds.PatientSex = ""  # (0010,0040)
        
        # 디버깅: PatientID와 PatientName 확인
        print(f"DICOM Slice {slice_idx + 1}: PatientID={ds.PatientID}, PatientName={ds.PatientName}")
        
        ds.StudyInstanceUID = study_instance_uid
        ds.StudyDate = datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.now().strftime("%H%M%S")
        ds.StudyID = str(uuid.uuid4())[:8]
        ds.StudyDescription = settings['study_description']
        # AccessionNumber: Study 식별 번호 (없으면 StudyID 기반 생성)
        if not hasattr(ds, 'AccessionNumber') or not ds.AccessionNumber:
            # StudyID를 기반으로 AccessionNumber 생성 (8자리)
            study_id = ds.StudyID if hasattr(ds, 'StudyID') else str(uuid.uuid4())[:8]
            ds.AccessionNumber = study_id
        ds.ReferringPhysicianName = ""  # Referring Physician Name
        
        ds.SeriesInstanceUID = series_instance_uid
        ds.SeriesNumber = str(series_number)  # 정수형을 문자열로 변환 (DICOM IS 타입)
        ds.SeriesDescription = settings['series_description']  # 영상 유형별 Description
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
        
        # 슬라이스 위치 및 Spacing 정보 (affine 행렬에서 추출) - 조원 코드 방식 적용
        try:
            if affine is not None and hasattr(affine, 'shape') and affine.shape == (4, 4):
                # PixelSpacing 계산 (affine 행렬에서)
                # DICOM 표준: [row_spacing, col_spacing] = [y, x] 순서
                pixel_spacing_x = np.sqrt(affine[0, 0]**2 + affine[1, 0]**2 + affine[2, 0]**2)
                pixel_spacing_y = np.sqrt(affine[0, 1]**2 + affine[1, 1]**2 + affine[2, 1]**2)
                pixel_spacing = [float(pixel_spacing_y), float(pixel_spacing_x)]  # [row, col] 순서
                
                # SliceThickness 계산 (z 방향)
                slice_thickness = np.sqrt(affine[0, 2]**2 + affine[1, 2]**2 + affine[2, 2]**2)
                
                # ImageOrientationPatient 계산 (affine 행렬에서 실제 방향 벡터 추출)
                # 첫 3개: row 방향, 다음 3개: column 방향
                row_direction = affine[:3, 1] / pixel_spacing_y if pixel_spacing_y > 0 else affine[:3, 1]
                col_direction = affine[:3, 0] / pixel_spacing_x if pixel_spacing_x > 0 else affine[:3, 0]
                image_orientation = [
                    float(col_direction[0]), float(col_direction[1]), float(col_direction[2]),
                    float(row_direction[0]), float(row_direction[1]), float(row_direction[2])
                ]
                
                # ImagePositionPatient 계산 (각 슬라이스의 3D 위치)
                position_homogeneous = affine @ np.array([0, 0, slice_idx, 1])
                image_position = [
                    float(position_homogeneous[0]),
                    float(position_homogeneous[1]),
                    float(position_homogeneous[2])
                ]
                
                # SliceLocation 계산
                slice_location = float(slice_idx * slice_thickness)
                
                # header에서 pixdim 확인 (우선순위 - 더 정확할 수 있음)
                if hasattr(header, 'get'):
                    pixdim = header.get('pixdim', [1, 1, 1, 1])
                    if len(pixdim) >= 4 and pixdim[1] > 0 and pixdim[2] > 0 and pixdim[3] > 0:
                        # pixdim이 있으면 사용 (더 정확할 수 있음)
                        pixel_spacing = [float(pixdim[2]), float(pixdim[1])]  # [y, x] 순서
                        slice_thickness = float(pixdim[3])
                        # ImageOrientationPatient는 affine에서 계산한 값 유지
                
            else:
                # Fallback: 기본값
                pixel_spacing = [1.0, 1.0]
                slice_thickness = 1.0
                image_orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                slice_location = float(slice_idx * slice_thickness)
                image_position = [0.0, 0.0, slice_location]
                
        except Exception as e:
            logger.warning(f"Failed to extract spatial information from affine/header: {e}")
            # Fallback: 기본값
            pixel_spacing = [1.0, 1.0]
            slice_thickness = 1.0
            image_orientation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            slice_location = float(slice_idx * slice_thickness)
            image_position = [0.0, 0.0, slice_location]
        
        # DICOM 메타데이터 설정 - MONAI Invertd 필수!
        ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]  # [row, col]
        ds.SliceThickness = str(slice_thickness)
        ds.ImagePositionPatient = [str(image_position[0]), str(image_position[1]), str(image_position[2])]
        ds.ImageOrientationPatient = [str(image_orientation[0]), str(image_orientation[1]), str(image_orientation[2]),
                                      str(image_orientation[3]), str(image_orientation[4]), str(image_orientation[5])]
        ds.SliceLocation = str(slice_location)
        ds.FrameOfReferenceUID = frame_of_reference_uid
        
        # NIfTI 헤더 정보를 DICOM에 보존 (모든 슬라이스에 동일하게 적용)
        # ImageComments에 NIfTI 메타데이터 저장
        comments_parts = []
        
        if nifti_metadata.get('descrip'):
            comments_parts.append(f"NIfTI descrip: {nifti_metadata['descrip']}")
        
        if nifti_metadata.get('cal_min') is not None or nifti_metadata.get('cal_max') is not None:
            comments_parts.append(f"NIfTI cal_min={nifti_metadata.get('cal_min', 'N/A')}, cal_max={nifti_metadata.get('cal_max', 'N/A')}")
        
        if nifti_metadata.get('intent_code'):
            comments_parts.append(f"NIfTI intent_code={nifti_metadata['intent_code']}")
        
        if comments_parts:
            ds.ImageComments = "\n".join(comments_parts)[:10240]  # DICOM LT 타입 제한 (10240자)
        
        # 첫 슬라이스에만 SeriesDescription에도 추가
        if slice_idx == 0 and nifti_metadata.get('descrip'):
            original_desc = ds.SeriesDescription
            nifti_desc = nifti_metadata['descrip'][:30]  # 짧게 유지
            if original_desc and original_desc != settings['series_description']:
                ds.SeriesDescription = f"{original_desc} [{nifti_desc}]"
            elif nifti_desc:
                # 기존 설명이 기본값이면 NIfTI 정보 추가
                ds.SeriesDescription = f"{settings['series_description']} [{nifti_desc}]"
        
        # 메타데이터 검증 로그 (첫 슬라이스에만)
        if slice_idx == 0:
            logger.info(f"📋 DICOM 메타데이터 확인 (첫 슬라이스):")
            logger.info(f"  ✅ PixelSpacing: {ds.PixelSpacing}")
            logger.info(f"  ✅ SliceThickness: {ds.SliceThickness}")
            logger.info(f"  ✅ ImagePositionPatient: {ds.ImagePositionPatient}")
            logger.info(f"  ✅ ImageOrientationPatient: {ds.ImageOrientationPatient}")
            logger.info(f"  ✅ FrameOfReferenceUID: {ds.FrameOfReferenceUID}")
            logger.info(f"  ✅ AccessionNumber: '{ds.AccessionNumber}'")
            if nifti_metadata.get('descrip'):
                logger.info(f"  ✅ NIfTI descrip 보존: '{nifti_metadata['descrip'][:50]}...'")
            if nifti_metadata.get('cal_min') is not None or nifti_metadata.get('cal_max') is not None:
                logger.info(f"  ✅ NIfTI cal_min/cal_max 보존: {nifti_metadata.get('cal_min')}/{nifti_metadata.get('cal_max')}")
            if nifti_metadata.get('intent_code'):
                logger.info(f"  ✅ NIfTI intent_code 보존: {nifti_metadata['intent_code']}")
            if hasattr(ds, 'ImageComments') and ds.ImageComments:
                logger.info(f"  ✅ ImageComments: '{ds.ImageComments[:100]}...'")
        
        # 픽셀 데이터 (numpy 배열을 직접 할당)
        ds.PixelData = slice_data.tobytes()
        
        # 파일로 저장 (메모리)
        buffer = BytesIO()
        
        # DICOM File Meta Information 설정 (필수)
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9"
        # TransferSyntaxUID 필수 추가 (Explicit VR Little Endian)
        file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
        
        # 데이터셋에 파일 메타 정보 연결
        ds.file_meta = file_meta
        ds.is_implicit_VR = False  # Explicit VR
        ds.is_little_endian = True  # Little Endian
        
        pydicom.dcmwrite(buffer, ds, write_like_original=False)
        dicom_slices.append(buffer.getvalue())
    
    return dicom_slices


def pil_image_to_dicom(pil_image, patient_id=None, patient_name=None, series_description="Heatmap Image", modality="MG", orthanc_client=None, study_instance_uid=None):
    """
    PIL Image를 DICOM으로 변환
    
    Args:
        pil_image: PIL Image 객체
        patient_id: 환자 ID
        patient_name: 환자 이름
        series_description: Series 설명
        modality: Modality (기본값: MG - Mammography)
        orthanc_client: OrthancClient 인스턴스 (기존 Study 찾기용, 선택사항)
        study_instance_uid: 기존 StudyInstanceUID (제공되면 재사용)
    
    Returns:
        bytes: DICOM 파일의 바이트 데이터
    """
    import numpy as np
    
    # PIL Image를 numpy 배열로 변환 (컬러 이미지 유지)
    is_color = pil_image.mode in ('RGB', 'RGBA')
    
    if pil_image.mode == 'RGBA':
        # RGBA를 RGB로 변환 (알파 채널 제거)
        pil_image = pil_image.convert('RGB')
        img_array = np.array(pil_image)
    elif pil_image.mode == 'RGB':
        # RGB 이미지는 그대로 유지
        img_array = np.array(pil_image)
    else:
        # 그레이스케일 이미지
        img_array = np.array(pil_image)
        if len(img_array.shape) == 2:
            # 2D 그레이스케일을 3D로 확장 (H, W) -> (H, W, 1)
            img_array = img_array[:, :, np.newaxis]
    
    # 컬러 이미지인 경우 (H, W, 3) 형태
    if is_color and len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # RGB 이미지를 uint16으로 변환 (각 채널별로)
        if img_array.dtype != np.uint16:
            # 0-65535 범위로 스케일링 (각 채널별)
            if img_array.max() > 0:
                img_array = (img_array.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
            else:
                img_array = img_array.astype(np.uint16)
    else:
        # 그레이스케일 이미지 처리
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]  # 첫 번째 채널만 사용
        
        # uint16으로 변환
        if img_array.dtype != np.uint16:
            if img_array.max() > 0:
                img_array = (img_array.astype(np.float32) / img_array.max() * 65535).astype(np.uint16)
            else:
                img_array = img_array.astype(np.uint16)
    
    # 환자 정보 설정
    if patient_id is None:
        patient_id = "UNKNOWN"
    if patient_name is None:
        patient_name = patient_id
    
    # 기존 Study 찾기 (같은 환자의 기존 Study에 속하도록)
    if study_instance_uid is None and orthanc_client is not None and patient_id:
        try:
            existing_uid = orthanc_client.get_existing_study_instance_uid(patient_id)
            if existing_uid:
                study_instance_uid = existing_uid
                logger.info(f"기존 StudyInstanceUID 재사용: {study_instance_uid[:20]}... (patient_id: {patient_id})")
        except Exception as e:
            logger.warning(f"기존 StudyInstanceUID 찾기 실패, 새로 생성: {e}")
    
    # Study 정보
    if study_instance_uid is None:
        study_instance_uid = generate_uid()
        logger.info(f"새 StudyInstanceUID 생성: {study_instance_uid[:20]}... (patient_id: {patient_id})")
    
    # DICOM 데이터셋 생성
    ds = Dataset()
    
    # 필수 DICOM 태그
    ds.PatientID = str(patient_id)
    ds.PatientName = str(patient_name)
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    
    # Study 정보
    ds.StudyInstanceUID = study_instance_uid
    ds.StudyDate = datetime.now().strftime("%Y%m%d")
    ds.StudyTime = datetime.now().strftime("%H%M%S")
    ds.StudyID = str(uuid.uuid4())[:8]
    ds.StudyDescription = "Mammography Analysis"
    ds.AccessionNumber = ""  # Accession Number
    ds.ReferringPhysicianName = ""  # Referring Physician Name
    
    # Series 정보
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = "1"
    ds.SeriesDescription = series_description
    ds.Modality = modality
    
    # Instance 정보
    ds.InstanceNumber = "1"
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"  # Digital Mammography X-Ray Image Storage
    
    # 이미지 파라미터
    ds.Rows = img_array.shape[0]
    ds.Columns = img_array.shape[1]
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned
    
    # 컬러 이미지인 경우 RGB 설정
    if is_color and len(img_array.shape) == 3 and img_array.shape[2] == 3:
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.PlanarConfiguration = 0  # 0 = interleaved (RGBRGBRGB...)
        # 픽셀 데이터를 interleaved 형식으로 변환 (R, G, B 순서)
        # (H, W, 3) -> (H*W, 3) -> (H*W*3) 형태로 변환
        # 각 픽셀의 R, G, B 값이 연속적으로 배치되도록
        h, w = img_array.shape[:2]
        pixel_data = img_array.reshape(h * w, 3).astype(np.uint16)
        # uint16 배열을 바이트로 변환 (little-endian)
        pixel_data = pixel_data.tobytes()
        logger.info(f"✅ RGB 컬러 이미지 처리: shape={img_array.shape}, pixel_data size={len(pixel_data)} bytes")
    else:
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        # 그레이스케일 이미지
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]
        pixel_data = img_array.astype(np.uint16).tobytes()
        logger.info(f"✅ 그레이스케일 이미지 처리: shape={img_array.shape}, pixel_data size={len(pixel_data)} bytes")
    
    # 픽셀 데이터
    ds.PixelData = pixel_data
    
    # 파일로 저장 (메모리)
    buffer = BytesIO()
    
    # DICOM File Meta Information 설정
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    file_meta.ImplementationClassUID = "1.2.3.4.5.6.7.8.9"
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.1'  # Explicit VR Little Endian
    
    ds.file_meta = file_meta
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    
    pydicom.dcmwrite(buffer, ds, write_like_original=False)
    return buffer.getvalue()

