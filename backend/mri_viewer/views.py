from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .utils import (
    get_patient_mri_data, 
    load_nifti_file, 
    get_slice_from_volume,
    normalize_slice,
    create_overlay,
    numpy_to_base64,
    load_mri_series
)
import json
from pathlib import Path
import numpy as np
from lung_cancer.models import Patient as LungCancerPatient


@api_view(['GET'])
def get_patient_list(request):
    """
    사용 가능한 환자 목록 반환
    """
    try:
        data_root = Path("/Users/nogeon-u/Desktop/건양대_바이오메디컬/Django/mmm")
        
        # Check if data root exists
        if not data_root.exists():
            return Response({
                'success': True,
                'patients': [],
                'message': 'Data directory not found. Please check the path configuration.'
            })
        
        image_dir = data_root / "images"
        
        # Check if images directory exists
        if not image_dir.exists():
            return Response({
                'success': True,
                'patients': [],
                'message': 'Images directory not found.'
            })
        
        patients = []
        try:
            for patient_dir in image_dir.iterdir():
                if patient_dir.is_dir():
                    patient_id = patient_dir.name
                    
                    # 환자 정보 파일 읽기
                    info_file = patient_dir / "info.json"
                    if info_file.exists():
                        with open(info_file, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            patients.append({
                                'patient_id': patient_id,
                                'name': info.get('name', patient_id),
                                'age': info.get('age'),
                                'gender': info.get('gender')
                            })
                    else:
                        patients.append({
                            'patient_id': patient_id,
                            'name': patient_id
                        })
        except Exception as e:
            print(f"Error reading patient directories: {e}")
        
        return Response({
            'success': True,
            'patients': patients
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_patient_info(request, patient_id):
    """
    특정 환자의 정보 반환
    """
    try:
        mri_data = get_patient_mri_data(patient_id)
        
        if not mri_data:
            return Response({
                'success': False,
                'error': 'Patient not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # 볼륨 정보 수집
        volume_info = []
        for series_idx, series_data in enumerate(mri_data['series']):
            volume_info.append({
                'series_index': series_idx,
                'num_slices': series_data['volume'].shape[2] if len(series_data['volume'].shape) > 2 else 1,
                'shape': series_data['volume'].shape,
                'series_description': series_data.get('series_description', f'Series {series_idx}')
            })
        
        return Response({
            'success': True,
            'patient_id': patient_id,
            'name': mri_data['info'].get('name', patient_id),
            'age': mri_data['info'].get('age'),
            'gender': mri_data['info'].get('gender'),
            'num_series': len(mri_data['series']),
            'series': volume_info,
            'has_segmentation': mri_data['segmentation_expert'] is not None
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_mri_slice(request, patient_id):
    """
    특정 환자의 MRI 슬라이스 이미지 반환
    Orthanc DICOM (유방촬영술) 및 NIfTI 파일 (MRI) 모두 지원
    """
    try:
        from .orthanc_client import OrthancClient
        import base64
        
        series = int(request.GET.get('series', 0))
        slice_idx = int(request.GET.get('slice', 0))
        axis = request.GET.get('axis', 'axial')
        show_segmentation = request.GET.get('segmentation', 'false').lower() == 'true'
        
        # 먼저 Orthanc에서 환자 찾기 (유방촬영술 DICOM 데이터)
        try:
            orthanc = OrthancClient()
            orthanc_patient_id = orthanc.find_patient_by_patient_id(patient_id)
            
            if orthanc_patient_id:
                # Orthanc에서 환자 발견 - DICOM 이미지 반환
                patient_info = orthanc.get_patient_info(orthanc_patient_id)
                studies = patient_info.get('Studies', [])
                
                if not studies:
                    return Response({
                        'success': False,
                        'error': 'No studies found for patient'
                    }, status=status.HTTP_404_NOT_FOUND)
                
                # 첫 번째 Study의 Series 가져오기
                study_info = orthanc.get_study_info(studies[0])
                series_list = study_info.get('Series', [])
                
                if series >= len(series_list):
                    return Response({
                        'success': False,
                        'error': f'Series index {series} out of range (max: {len(series_list)-1})'
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # 선택된 Series의 Instances 가져오기
                series_info = orthanc.get_series_info(series_list[series])
                instances = series_info.get('Instances', [])
                
                if slice_idx >= len(instances):
                    return Response({
                        'success': False,
                        'error': f'Slice index {slice_idx} out of range (max: {len(instances)-1})'
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # Instance 미리보기 이미지 가져오기 (PNG)
                instance_id = instances[slice_idx]
                preview_image = orthanc.get_instance_preview(instance_id)
                
                # Base64 인코딩
                image_base64 = base64.b64encode(preview_image).decode('utf-8')
                
                return Response({
                    'success': True,
                    'image': f'data:image/png;base64,{image_base64}',
                    'slice': slice_idx,
                    'series': series,
                    'axis': axis,
                    'source': 'orthanc'
                })
        except Exception as orthanc_error:
            # Orthanc에서 찾지 못하면 NIfTI 파일 시스템으로 fallback
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Orthanc lookup failed for {patient_id}: {orthanc_error}")
        
        # NIfTI 파일 시스템에서 찾기 (기존 MRI 데이터)
        try:
            mri_data = get_patient_mri_data(patient_id)
        except FileNotFoundError:
            # Orthanc과 NIfTI 모두에서 찾지 못함
            return Response({
                'success': False,
                'error': f'Patient "{patient_id}" not found in Orthanc or file system. Please upload DICOM files to Orthanc first.',
                'suggestion': 'Check if the PatientID matches the DICOM files uploaded to Orthanc.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as nifti_error:
            # NIfTI 파일 로드 중 다른 에러 발생
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error loading NIfTI data for {patient_id}: {nifti_error}")
            return Response({
                'success': False,
                'error': f'Error loading patient data: {str(nifti_error)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        if not mri_data:
            return Response({
                'success': False,
                'error': 'Patient not found in both Orthanc and file system'
            }, status=status.HTTP_404_NOT_FOUND)
        
        if series >= len(mri_data['series']):
            return Response({
                'success': False,
                'error': 'Series index out of range'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # 볼륨 데이터 가져오기
        volume = mri_data['series'][series]['volume']
        
        # 슬라이스 추출
        slice_data = get_slice_from_volume(volume, slice_idx, axis)
        
        # 정규화
        slice_normalized = normalize_slice(slice_data)
        
        # 세그멘테이션 오버레이
        if show_segmentation and mri_data['segmentation_expert'] is not None:
            seg_slice = get_slice_from_volume(mri_data['segmentation_expert'], slice_idx, axis)
            slice_normalized = create_overlay(slice_normalized, seg_slice)
        
        # Base64 인코딩
        image_base64 = numpy_to_base64(slice_normalized)
        
        return Response({
            'success': True,
            'image': image_base64,
            'slice': slice_idx,
            'series': series,
            'axis': axis,
            'source': 'nifti'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_volume_info(request, patient_id):
    """
    환자의 전체 볼륨 정보 반환
    """
    try:
        mri_data = get_patient_mri_data(patient_id)
        
        if not mri_data:
            return Response({
                'success': False,
                'error': 'Patient not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # 시리즈별 볼륨 정보
        volume_info = []
        for series_idx, series_data in enumerate(mri_data['series']):
            volume = series_data['volume']
            volume_info.append({
                'series_index': series_idx,
                'shape': volume.shape,
                'num_slices': volume.shape[2] if len(volume.shape) > 2 else 1,
                'series_description': series_data.get('series_description', f'Series {series_idx}'),
                'min_value': float(np.min(volume)),
                'max_value': float(np.max(volume)),
                'mean_value': float(np.mean(volume))
            })
        
        return Response({
            'success': True,
            'patient_id': patient_id,
            'series': volume_info,
            'has_segmentation': mri_data['segmentation_expert'] is not None,
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_patient_basic_info(request, patient_id):
    """
    환자 기본 정보 조회 (lung_cancer 앱에서)
    404 에러 방지를 위한 엔드포인트
    """
    try:
        # lung_cancer 앱에서 환자 정보 조회
        patient = LungCancerPatient.objects.filter(patient_id=patient_id).first()
        
        if patient:
            return Response({
                'success': True,
                'patient_id': patient.patient_id,
                'name': patient.name,
                'phone': patient.phone,
                'gender': patient.gender,
                'age': patient.age,
                'birth_date': patient.birth_date.isoformat() if patient.birth_date else None,
            })
        else:
            # 환자가 없어도 404 대신 빈 응답 반환
            return Response({
                'success': True,
                'patient_id': patient_id,
                'name': 'Unknown',
                'message': 'Patient not found in system'
            })
            
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
