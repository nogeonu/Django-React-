"""
Orthanc PACS 서버 연동 API Views
"""
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.http import HttpResponse
from .orthanc_client import OrthancClient
import traceback


@api_view(['GET'])
def orthanc_system_info(request):
    """Orthanc 시스템 정보"""
    try:
        client = OrthancClient()
        system_info = client.get_system_info()
        statistics = client.get_statistics()
        
        return Response({
            'success': True,
            'system': system_info,
            'statistics': statistics
        })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def orthanc_patients(request):
    """Orthanc 환자 목록"""
    try:
        client = OrthancClient()
        patient_ids = client.get_patients()
        
        patients = []
        for patient_id in patient_ids:
            try:
                info = client.get_patient_info(patient_id)
                patients.append({
                    'id': patient_id,
                    'main_dicom_tags': info.get('MainDicomTags', {}),
                    'studies': info.get('Studies', []),
                    'type': info.get('Type', ''),
                })
            except Exception as e:
                print(f"Error getting patient {patient_id}: {e}")
                continue
        
        return Response({
            'success': True,
            'patients': patients,
            'count': len(patients)
        })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def orthanc_patient_detail(request, patient_id):
    """환자 상세 정보 및 이미지"""
    try:
        client = OrthancClient()
        patient_info = client.get_patient_info(patient_id)
        studies = client.get_patient_studies(patient_id)
        
        images = []
        for study_id in studies:
            study_info = client.get_study_info(study_id)
            series_list = client.get_study_series(study_id)
            
            for series_id in series_list:
                series_info = client.get_series_info(series_id)
                instances = client.get_series_instances(series_id)
                
                for instance_id in instances:
                    instance_info = client.get_instance_info(instance_id)
                    images.append({
                        'instance_id': instance_id,
                        'series_id': series_id,
                        'study_id': study_id,
                        'series_description': series_info.get('MainDicomTags', {}).get('SeriesDescription', ''),
                        'instance_number': instance_info.get('MainDicomTags', {}).get('InstanceNumber', ''),
                        'preview_url': f'/api/mri/orthanc/instances/{instance_id}/preview/',
                    })
        
        return Response({
            'success': True,
            'patient': patient_info,
            'images': images,
            'image_count': len(images)
        })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def orthanc_instance_preview(request, instance_id):
    """Instance 미리보기 이미지 (PNG)"""
    try:
        client = OrthancClient()
        image_data = client.get_instance_preview(instance_id)
        return HttpResponse(image_data, content_type='image/png')
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def orthanc_upload_dicom(request):
    """DICOM 또는 NIfTI 파일 업로드"""
    try:
        # 디버깅 로그
        print(f"Request method: {request.method}")
        print(f"Request FILES keys: {list(request.FILES.keys())}")
        print(f"Request data keys: {list(request.data.keys())}")
        
        if 'file' not in request.FILES:
            return Response({
                'success': False,
                'error': f'No file provided. Available keys: {list(request.FILES.keys())}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        file_name = uploaded_file.name.lower()
        file_size = uploaded_file.size
        patient_id = request.data.get('patient_id', None)
        
        print(f"Uploaded file: {file_name}, size: {file_size} bytes, patient_id: {patient_id}")
        
        client = OrthancClient()
        
        # 파일 확장자 확인
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            # NIfTI 파일인 경우 DICOM으로 변환
            try:
                from .utils import nifti_to_dicom_slices
                from io import BytesIO
                
                # 파일을 메모리로 읽기
                file_data = uploaded_file.read()
                
                if len(file_data) == 0:
                    return Response({
                        'success': False,
                        'error': 'Uploaded file is empty'
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # NIfTI를 DICOM 슬라이스들로 변환
                try:
                    dicom_slices = nifti_to_dicom_slices(
                        BytesIO(file_data),
                        patient_id=patient_id or "UNKNOWN",
                        patient_name=patient_id or "UNKNOWN"
                    )
                except Exception as e:
                    return Response({
                        'success': False,
                        'error': f'NIfTI 파일 변환 실패: {str(e)}',
                        'traceback': traceback.format_exc()
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                if not dicom_slices or len(dicom_slices) == 0:
                    return Response({
                        'success': False,
                        'error': 'DICOM 슬라이스 변환 결과가 비어있습니다.'
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # 각 DICOM 슬라이스를 Orthanc에 업로드
                uploaded_count = 0
                errors = []
                for idx, dicom_slice in enumerate(dicom_slices):
                    try:
                        result = client.upload_dicom(dicom_slice)
                        uploaded_count += 1
                    except Exception as e:
                        error_msg = f"슬라이스 {idx+1} 업로드 실패: {str(e)}"
                        errors.append(error_msg)
                        print(error_msg)
                        continue
                
                if uploaded_count == 0:
                    return Response({
                        'success': False,
                        'error': '모든 슬라이스 업로드 실패',
                        'errors': errors
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
                return Response({
                    'success': True,
                    'message': f'NIfTI 파일이 변환되어 업로드되었습니다. {uploaded_count}/{len(dicom_slices)} 슬라이스 업로드 완료.',
                    'slices_uploaded': uploaded_count,
                    'total_slices': len(dicom_slices),
                    'errors': errors if errors else None
                })
            except Exception as e:
                return Response({
                    'success': False,
                    'error': f'NIfTI 파일 처리 중 오류: {str(e)}',
                    'traceback': traceback.format_exc()
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # DICOM 파일인 경우 직접 업로드
            dicom_data = uploaded_file.read()
            result = client.upload_dicom(dicom_data)
            
            return Response({
                'success': True,
                'result': result,
                'message': 'DICOM file uploaded successfully'
            })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['DELETE'])
def orthanc_delete_patient(request, patient_id):
    """환자 데이터 삭제"""
    try:
        client = OrthancClient()
        client.delete_patient(patient_id)
        
        return Response({
            'success': True,
            'message': f'Patient {patient_id} deleted successfully'
        })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

