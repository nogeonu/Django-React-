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
                    info_file = data_root / "patient_info_files" / f"{patient_id.lower()}.json"
                    patient_info = {}
                    if info_file.exists():
                        try:
                            with open(info_file, 'r', encoding='utf-8') as f:
                                patient_info = json.load(f)
                        except Exception as json_error:
                            print(f"Error reading patient info for {patient_id}: {json_error}")
                    
                    patients.append({
                        'patient_id': patient_id,
                        'age': patient_info.get('clinical_data', {}).get('age'),
                        'tumor_subtype': patient_info.get('primary_lesion', {}).get('tumor_subtype'),
                        'scanner_manufacturer': patient_info.get('imaging_data', {}).get('scanner_manufacturer'),
                    })
        except PermissionError as pe:
            return Response({
                'success': True,
                'patients': [],
                'message': f'Permission denied accessing patient data: {str(pe)}'
            })
        
        return Response({
            'success': True,
            'patients': patients
        })
    except Exception as e:
        import traceback
        print(f"Error in get_patient_list: {str(e)}")
        print(traceback.format_exc())
        return Response({
            'success': False,
            'error': str(e),
            'details': 'An error occurred while fetching patient list. Please check server logs.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_patient_info(request, patient_id):
    """
    특정 환자의 MRI 정보 반환
    """
    try:
        try:
            mri_data = get_patient_mri_data(patient_id)
        except FileNotFoundError as fnf:
            return Response({
                'success': False,
                'error': str(fnf)
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as data_error:
            return Response({
                'success': False,
                'error': f'Error loading patient data: {str(data_error)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # 환자 정보 파일 읽기
        patient_info = {}
        if mri_data['patient_info_file']:
            try:
                with open(mri_data['patient_info_file'], 'r', encoding='utf-8') as f:
                    patient_info = json.load(f)
            except Exception as info_error:
                print(f"Error reading patient info file: {info_error}")
        
        # 첫 번째 이미지 파일의 메타데이터 읽기
        if mri_data['image_files']:
            try:
                first_image = mri_data['image_files'][0]
                data, affine, header = load_nifti_file(first_image)
                
                return Response({
                    'success': True,
                    'patient_id': patient_id,
                    'patient_info': patient_info,
                    'series': [
                        {
                            'filename': Path(f).name,
                            'index': i
                        } for i, f in enumerate(mri_data['image_files'])
                    ],
                    'has_segmentation': mri_data['segmentation_expert'] is not None,
                    'volume_shape': data.shape,
                    'num_slices': data.shape[2] if len(data.shape) > 2 else 1,
                })
            except Exception as nifti_error:
                return Response({
                    'success': False,
                    'error': f'Error loading NIfTI file: {str(nifti_error)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({
                'success': False,
                'error': '이미지 파일을 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
            
    except Exception as e:
        import traceback
        print(f"Error in get_patient_info: {str(e)}")
        print(traceback.format_exc())
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_mri_slice(request, patient_id):
    """
    MRI 슬라이스 이미지 반환
    """
    try:
        # 쿼리 파라미터
        series_idx = int(request.GET.get('series', 0))
        slice_idx = int(request.GET.get('slice', 0))
        axis = request.GET.get('axis', 'axial')
        show_segmentation = request.GET.get('segmentation', 'false').lower() == 'true'
        
        # 데이터 로드
        mri_data = get_patient_mri_data(patient_id)
        
        if series_idx >= len(mri_data['image_files']):
            return Response({
                'success': False,
                'error': '잘못된 시리즈 인덱스입니다.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # 이미지 로드
        image_file = mri_data['image_files'][series_idx]
        image_data, _, _ = load_nifti_file(image_file)
        
        # 슬라이스 추출
        image_slice = get_slice_from_volume(image_data, slice_idx, axis)
        image_slice = normalize_slice(image_slice)
        
        # 세그멘테이션 오버레이
        if show_segmentation and mri_data['segmentation_expert']:
            seg_data, _, _ = load_nifti_file(mri_data['segmentation_expert'])
            seg_slice = get_slice_from_volume(seg_data, slice_idx, axis)
            seg_slice = normalize_slice(seg_slice)
            
            # 오버레이 생성
            overlay_image = create_overlay(image_slice, seg_slice, alpha=0.3)
            image_base64 = numpy_to_base64(overlay_image)
        else:
            # 그레이스케일 이미지를 RGB로 변환
            image_rgb = np.stack([image_slice] * 3, axis=-1)
            image_base64 = numpy_to_base64(image_rgb)
        
        return Response({
            'success': True,
            'image': image_base64,
            'slice_index': slice_idx,
            'series_index': series_idx,
            'axis': axis,
            'shape': image_slice.shape,
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_volume_info(request, patient_id):
    """
    전체 볼륨 정보 반환 (모든 시리즈)
    """
    try:
        mri_data = get_patient_mri_data(patient_id)
        series_list = load_mri_series(mri_data['image_files'])
        
        volume_info = []
        for i, series in enumerate(series_list):
            volume_info.append({
                'index': i,
                'filename': series['filename'],
                'shape': series['shape'],
                'num_slices': series['shape'][2] if len(series['shape']) > 2 else 1,
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

