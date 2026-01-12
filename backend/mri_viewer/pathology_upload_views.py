"""
병리 이미지(SVS) 업로드 및 Orthanc 저장 API
"""
import os
import io
import logging
import pydicom
import numpy as np
from datetime import datetime
from PIL import Image
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import requests

logger = logging.getLogger(__name__)

# Orthanc 설정
ORTHANC_URL = os.getenv('ORTHANC_URL', 'http://localhost:8042')
ORTHANC_USERNAME = os.getenv('ORTHANC_USERNAME', 'admin')
ORTHANC_PASSWORD = os.getenv('ORTHANC_PASSWORD', 'admin123')


def svs_to_dicom(svs_file, patient_id, patient_name, study_description="Pathology WSI"):
    """
    SVS 파일을 DICOM으로 변환
    
    Args:
        svs_file: SVS 파일 객체
        patient_id: 환자 ID
        patient_name: 환자 이름
        study_description: 검사 설명
    
    Returns:
        DICOM 파일 바이트
    """
    try:
        import openslide
        
        # 임시 파일로 저장 (OpenSlide는 파일 경로 필요)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.svs', delete=False) as tmp_file:
            for chunk in svs_file.chunks():
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        try:
            # SVS 파일 열기
            slide = openslide.OpenSlide(tmp_path)
            
            # 썸네일 생성 (512x512)
            thumbnail_size = (512, 512)
            thumbnail = slide.get_thumbnail(thumbnail_size)
            thumbnail_array = np.array(thumbnail)
            
            # RGB 컬러 유지 (병리 이미지는 염색 정보가 중요)
            if len(thumbnail_array.shape) == 3 and thumbnail_array.shape[2] >= 3:
                # RGB 이미지 유지 (알파 채널 제거)
                thumbnail_rgb = thumbnail_array[:, :, :3].astype(np.uint8)
            else:
                # 흑백 이미지를 RGB로 변환
                thumbnail_rgb = np.stack([thumbnail_array] * 3, axis=-1).astype(np.uint8)
            
            logger.info(f"🎨 썸네일 생성: {thumbnail_rgb.shape}, dtype={thumbnail_rgb.dtype}")
            
            # DICOM 파일 생성
            file_meta = pydicom.dataset.FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'  # VL Whole Slide Microscopy Image
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
            
            ds = pydicom.dataset.FileDataset(
                None, {}, 
                file_meta=file_meta, 
                preamble=b"\0" * 128
            )
            
            # Patient Information
            ds.PatientName = patient_name
            ds.PatientID = patient_id
            ds.PatientBirthDate = ''
            ds.PatientSex = ''
            
            # Study Information
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
            ds.StudyDate = datetime.now().strftime('%Y%m%d')
            ds.StudyTime = datetime.now().strftime('%H%M%S')
            ds.StudyDescription = study_description
            ds.StudyID = '1'
            
            # Series Information
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesNumber = '1'
            ds.SeriesDescription = f"Pathology WSI - {svs_file.name}"
            ds.Modality = 'SM'  # Slide Microscopy
            
            # Instance Information
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.InstanceNumber = '1'
            
            # Image Information (RGB 컬러)
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = 'RGB'
            ds.PlanarConfiguration = 0  # 0 = R1G1B1R2G2B2... (인터리브)
            ds.Rows = thumbnail_rgb.shape[0]
            ds.Columns = thumbnail_rgb.shape[1]
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            
            # Pixel Data (RGB)
            ds.PixelData = thumbnail_rgb.tobytes()
            
            # WSI 메타데이터 추가
            ds.TotalPixelMatrixColumns = slide.dimensions[0]
            ds.TotalPixelMatrixRows = slide.dimensions[1]
            
            # DICOM 파일을 바이트로 변환
            output = io.BytesIO()
            ds.save_as(output, write_like_original=False)
            dicom_bytes = output.getvalue()
            
            logger.info(f"✅ SVS to DICOM 변환 완료: {svs_file.name} -> {len(dicom_bytes)} bytes")
            
            return dicom_bytes
            
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_path)
            slide.close()
            
    except Exception as e:
        logger.error(f"❌ SVS to DICOM 변환 오류: {str(e)}", exc_info=True)
        raise


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def upload_pathology_image(request):
    """
    병리 이미지(SVS) 업로드 및 Orthanc 저장
    
    Request:
        - file: SVS 파일
        - patient_id: 환자 ID
        - patient_name: 환자 이름 (선택)
        - study_description: 검사 설명 (선택)
    
    Response:
        {
            "success": true,
            "patient_id": "P12345",
            "orthanc_patient_id": "...",
            "study_id": "...",
            "series_id": "...",
            "instance_id": "..."
        }
    """
    try:
        # 파일 확인
        if 'file' not in request.FILES:
            return Response(
                {'error': '파일이 필요합니다'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        svs_file = request.FILES['file']
        
        # SVS 파일 확인
        if not svs_file.name.lower().endswith('.svs'):
            return Response(
                {'error': 'SVS 파일만 업로드 가능합니다'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 환자 정보
        patient_id = request.data.get('patient_id', f'PATH_{datetime.now().strftime("%Y%m%d%H%M%S")}')
        patient_name = request.data.get('patient_name', patient_id)
        study_description = request.data.get('study_description', 'Pathology WSI')
        
        logger.info(f"📥 병리 이미지 업로드 시작: {svs_file.name}")
        logger.info(f"👤 환자 ID: {patient_id}, 이름: {patient_name}")
        
        # SVS를 DICOM으로 변환
        logger.info(f"🔄 SVS to DICOM 변환 중...")
        dicom_bytes = svs_to_dicom(svs_file, patient_id, patient_name, study_description)
        
        # Orthanc에 업로드
        logger.info(f"📤 Orthanc에 업로드 중...")
        response = requests.post(
            f"{ORTHANC_URL}/instances",
            data=dicom_bytes,
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
            headers={'Content-Type': 'application/dicom'},
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Orthanc 업로드 실패: {response.status_code} - {response.text}")
            return Response(
                {'error': f'Orthanc 업로드 실패: {response.status_code}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        result = response.json()
        
        logger.info(f"✅ 병리 이미지 업로드 완료")
        logger.info(f"📊 Patient: {result.get('ParentPatient')}")
        logger.info(f"📊 Study: {result.get('ParentStudy')}")
        logger.info(f"📊 Series: {result.get('ParentSeries')}")
        logger.info(f"📊 Instance: {result.get('ID')}")
        
        return Response({
            'success': True,
            'patient_id': patient_id,
            'orthanc_patient_id': result.get('ParentPatient'),
            'study_id': result.get('ParentStudy'),
            'series_id': result.get('ParentSeries'),
            'instance_id': result.get('ID'),
            'file_name': svs_file.name
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.error(f"❌ 병리 이미지 업로드 오류: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_pathology_images(request):
    """
    Orthanc에 저장된 병리 이미지 목록 조회
    
    Query Parameters:
        - patient_id: 환자 ID (선택)
    
    Response:
        {
            "success": true,
            "images": [
                {
                    "patient_id": "...",
                    "patient_name": "...",
                    "study_id": "...",
                    "series_id": "...",
                    "instance_id": "...",
                    "study_date": "...",
                    "series_description": "..."
                }
            ]
        }
    """
    try:
        patient_id_filter = request.query_params.get('patient_id')
        
        # Orthanc에서 모든 환자 조회
        response = requests.get(
            f"{ORTHANC_URL}/patients",
            auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
            timeout=30
        )
        response.raise_for_status()
        patient_ids = response.json()
        
        images = []
        
        for orthanc_patient_id in patient_ids:
            # 환자 정보 조회
            patient_response = requests.get(
                f"{ORTHANC_URL}/patients/{orthanc_patient_id}",
                auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
                timeout=30
            )
            patient_info = patient_response.json()
            patient_tags = patient_info.get('MainDicomTags', {})
            patient_id = patient_tags.get('PatientID', '')
            
            # 필터링
            if patient_id_filter and patient_id != patient_id_filter:
                continue
            
            # 스터디 조회
            for study_id in patient_info.get('Studies', []):
                study_response = requests.get(
                    f"{ORTHANC_URL}/studies/{study_id}",
                    auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
                    timeout=30
                )
                study_info = study_response.json()
                study_tags = study_info.get('MainDicomTags', {})
                
                # Modality가 SM (Slide Microscopy)인 경우만
                for series_id in study_info.get('Series', []):
                    series_response = requests.get(
                        f"{ORTHANC_URL}/series/{series_id}",
                        auth=(ORTHANC_USERNAME, ORTHANC_PASSWORD),
                        timeout=30
                    )
                    series_info = series_response.json()
                    series_tags = series_info.get('MainDicomTags', {})
                    
                    if series_tags.get('Modality') == 'SM':
                        for instance_id in series_info.get('Instances', []):
                            images.append({
                                'patient_id': patient_id,
                                'patient_name': patient_tags.get('PatientName', ''),
                                'orthanc_patient_id': orthanc_patient_id,
                                'study_id': study_id,
                                'series_id': series_id,
                                'instance_id': instance_id,
                                'study_date': study_tags.get('StudyDate', ''),
                                'study_description': study_tags.get('StudyDescription', ''),
                                'series_description': series_tags.get('SeriesDescription', '')
                            })
        
        return Response({
            'success': True,
            'total': len(images),
            'images': images
        })
        
    except Exception as e:
        logger.error(f"❌ 병리 이미지 목록 조회 오류: {str(e)}", exc_info=True)
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

