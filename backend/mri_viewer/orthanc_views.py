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
import requests


@api_view(['GET'])
def orthanc_debug_patients(request):
    """디버깅용: 모든 환자와 그들의 PatientID 목록"""
    try:
        client = OrthancClient()
        all_patients = client.get_patients()
        
        patient_list = []
        for orthanc_id in all_patients:
            try:
                response = requests.get(f"{client.base_url}/patients/{orthanc_id}", auth=client.auth)
                response.raise_for_status()
                info = response.json()
                tags = info.get('MainDicomTags', {})
                patient_list.append({
                    'orthanc_id': orthanc_id,
                    'dicom_patient_id': tags.get('PatientID', 'N/A'),
                    'patient_name': tags.get('PatientName', 'N/A'),
                })
            except:
                patient_list.append({
                    'orthanc_id': orthanc_id,
                    'dicom_patient_id': 'ERROR',
                    'patient_name': 'ERROR',
                })
        
        return Response({
            'success': True,
            'total_patients': len(all_patients),
            'patients': patient_list
        })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


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
    """환자 상세 정보 및 이미지 (DICOM PatientID 또는 Orthanc 내부 ID 사용)"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        client = OrthancClient()
        logger.debug(f"Looking up patient with ID: {patient_id}")
        
        # 먼저 DICOM PatientID 태그로 환자 찾기 시도
        logger.info(f"=== Starting patient lookup for PatientID: '{patient_id}' ===")
        orthanc_patient_id = client.find_patient_by_patient_id(patient_id)
        logger.info(f"find_patient_by_patient_id result: {orthanc_patient_id}")
        
        # 찾지 못했으면 직접 접근 시도 (Orthanc 내부 ID일 수 있음)
        if not orthanc_patient_id:
            try:
                # 직접 접근 시도
                response = requests.get(
                    f"{client.base_url}/patients/{patient_id}",
                    auth=client.auth
                )
                response.raise_for_status()
                orthanc_patient_id = patient_id
                logger.debug(f"Direct access succeeded, using patient_id as orthanc_patient_id: {orthanc_patient_id}")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # 찾을 수 없음 - 모든 환자 목록과 PatientID를 로깅 (더 자세히)
                    try:
                        all_patients = client.get_patients()
                        logger.error(f"Patient '{patient_id}' not found after all attempts. Total patients in Orthanc: {len(all_patients)}")
                        logger.error(f"Available Orthanc patient IDs (first 10): {all_patients[:10]}")
                        
                        # 모든 환자의 실제 PatientID 태그 확인해서 로깅 (제한 없이 모두 확인)
                        logger.error("=== Searching all stored DICOM PatientIDs ===")
                        for idx, pid in enumerate(all_patients):
                            try:
                                pat_info = requests.get(f"{client.base_url}/patients/{pid}", auth=client.auth).json()
                                actual_pid = pat_info.get('MainDicomTags', {}).get('PatientID', 'N/A')
                                patient_name = pat_info.get('MainDicomTags', {}).get('PatientName', 'N/A')
                                
                                # 처음 10개만 상세 로깅, 나머지는 간단히
                                if idx < 10:
                                    logger.error(f"  [{idx+1}] Orthanc ID: {pid} -> DICOM PatientID: '{actual_pid}' | PatientName: '{patient_name}'")
                                
                                # 정확히 일치하는지 먼저 확인
                                if actual_pid == patient_id:
                                    logger.error(f"      -> EXACT MATCH FOUND! Orthanc ID: {pid} for PatientID '{patient_id}'")
                                    orthanc_patient_id = pid
                                    logger.error(f"      -> Using this as orthanc_patient_id: {orthanc_patient_id}")
                                    break
                                # 대소문자 무시하고 일치하는지 확인
                                elif actual_pid and actual_pid.strip().upper() == patient_id.strip().upper():
                                    logger.error(f"      -> CASE-INSENSITIVE MATCH FOUND! Orthanc ID: {pid} (stored: '{actual_pid}', searched: '{patient_id}')")
                                    orthanc_patient_id = pid
                                    logger.error(f"      -> Using this as orthanc_patient_id: {orthanc_patient_id}")
                                    break
                            except Exception as pid_error:
                                logger.debug(f"Error checking patient {pid}: {pid_error}")
                                continue  # 제한 없이 계속 확인
                        
                        if orthanc_patient_id:
                            logger.error(f"=== Found patient via fallback: {orthanc_patient_id} ===")
                        else:
                            logger.error("=== End of PatientID list - NO MATCH FOUND ===")
                    except Exception as log_error:
                        logger.error(f"Error while logging patient list: {log_error}", exc_info=True)
                    
                    # fallback에서 찾았는지 확인
                    if not orthanc_patient_id:
                        return Response({
                            'success': False,
                            'error': f'Patient ID "{patient_id}" not found in Orthanc. Please check if the DICOM file was uploaded with this PatientID.',
                            'suggestion': 'Upload a DICOM file with PatientID matching this patient first.',
                            'debug': 'Check server logs for detailed patient list'
                        }, status=status.HTTP_404_NOT_FOUND)
                    else:
                        logger.info(f"Found patient via fallback iteration: {orthanc_patient_id}")
                else:
                    # 404가 아닌 다른 HTTP 에러인 경우
                    raise
        
        logger.debug(f"Using orthanc_patient_id: {orthanc_patient_id}")
        
        # Orthanc 내부 ID로 직접 정보 가져오기 (get_patient_info는 재검색을 시도할 수 있으므로 직접 호출)
        try:
            response = requests.get(f"{client.base_url}/patients/{orthanc_patient_id}", auth=client.auth)
            response.raise_for_status()
            patient_info = response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to get patient info for orthanc_patient_id {orthanc_patient_id}: {e}")
            raise
        
        studies = client.get_patient_studies(orthanc_patient_id)
        logger.debug(f"Found {len(studies)} studies for patient {orthanc_patient_id}")
        
        images = []
        for study in studies:
            # Extract ID from dict if needed (Orthanc returns list of dicts)
            study_id = study if isinstance(study, str) else study.get('ID')
            logger.debug(f"Processing study: {study_id}")
            
            study_info = client.get_study_info(study_id)
            series_list = client.get_study_series(study_id)
            
            for series in series_list:
                # Extract ID from dict if needed
                series_id = series if isinstance(series, str) else series.get('ID')
                logger.debug(f"Processing series: {series_id}")
                
                series_info = client.get_series_info(series_id)
                instances = client.get_series_instances(series_id)
                
                for instance in instances:
                    # Extract ID from dict if needed
                    instance_id = instance if isinstance(instance, str) else instance.get('ID')
                    
                    instance_info = client.get_instance_info(instance_id)
                    instance_tags = instance_info.get('MainDicomTags', {})
                    
                    # 유방촬영술을 위한 추가 태그 수집
                    view_position = instance_tags.get('ViewPosition', '')  # CC, MLO 등
                    image_laterality = instance_tags.get('ImageLaterality', '')  # L, R
                    
                    # ViewPosition과 ImageLaterality를 조합하여 mammography_view 생성
                    # 예: L + CC = LCC, R + MLO = RMLO
                    mammography_view = ''
                    if view_position and image_laterality:
                        mammography_view = f"{image_laterality}{view_position}"  # LCC, RCC, LMLO, RMLO
                    
                    images.append({
                        'instance_id': instance_id,
                        'series_id': series_id,
                        'study_id': study_id,
                        'series_description': series_info.get('MainDicomTags', {}).get('SeriesDescription', ''),
                        'instance_number': instance_tags.get('InstanceNumber', ''),
                        'preview_url': f'/api/mri/orthanc/instances/{instance_id}/preview/',
                        'view_position': view_position,  # CC, MLO
                        'image_laterality': image_laterality,  # L, R
                        'mammography_view': mammography_view,  # LCC, RCC, LMLO, RMLO
                    })
        
        logger.debug(f"Returning {len(images)} images for patient {orthanc_patient_id}")
        return Response({
            'success': True,
            'patient': patient_info,
            'images': images,
            'image_count': len(images),
            'orthanc_patient_id': orthanc_patient_id  # 디버깅용
        })
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error while fetching patient detail for {patient_id}: {e}", exc_info=True)
        if e.response.status_code == 404:
            return Response({
                'success': False,
                'error': f'Patient ID "{patient_id}" not found in Orthanc.',
                'traceback': traceback.format_exc()
            }, status=status.HTTP_404_NOT_FOUND)
        return Response({
            'success': False,
            'error': f'Orthanc API error: {str(e)}',
            'traceback': traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except Exception as e:
        logger.error(f"Error fetching patient detail for {patient_id}: {e}", exc_info=True)
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


@api_view(['GET'])
def orthanc_instance_file(request, instance_id):
    """Instance DICOM 파일 다운로드 (Cornerstone3D용)"""
    try:
        client = OrthancClient()
        dicom_data = client.get_instance_file(instance_id)
        return HttpResponse(dicom_data, content_type='application/dicom')
    except Exception as e:
        logger.error(f"Failed to get DICOM file for instance {instance_id}: {e}")
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
        
        if 'file' not in request.FILES:
            return Response({
                'success': False,
                'error': f'No file provided.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        file_name = uploaded_file.name.lower()
        patient_id = request.data.get('patient_id', None)
        
        # 환자 이름 조회
        patient_name = "UNKNOWN"
        if patient_id:
            try:
                from patients.models import Patient
                patient = Patient.objects.filter(patient_id=patient_id).first()
                if patient:
                    patient_name = patient.name
                    print(f"Found patient name for {patient_id}: {patient_name}")
                else:
                    print(f"Patient ID {patient_id} not found in database, using ID as name")
                    patient_name = patient_id
            except Exception as e:
                print(f"Error fetching patient name: {e}")
                patient_name = patient_id

        print(f"Uploading file: {file_name}, patient: {patient_name} ({patient_id})")
        
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
                        patient_name=patient_name  # DB에서 찾은 이름 전달
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
                uploaded_instance_ids = []  # 업로드된 인스턴스 ID 저장
                for idx, dicom_slice in enumerate(dicom_slices):
                    try:
                        result = client.upload_dicom(dicom_slice)
                        uploaded_count += 1
                        if 'ID' in result:
                            uploaded_instance_ids.append(result['ID'])
                    except Exception as e:
                        error_msg = f"슬라이스 {idx+1} 업로드 실패: {str(e)}"
                        errors.append(error_msg)
                        print(error_msg)
                        continue
                
                # 업로드된 첫 번째 인스턴스의 PatientID 확인
                actual_patient_id = patient_id or "UNKNOWN"
                if uploaded_instance_ids:
                    try:
                        first_instance = client.get_instance_info(uploaded_instance_ids[0])
                        tags = first_instance.get('MainDicomTags', {})
                        actual_patient_id = tags.get('PatientID', patient_id or "UNKNOWN")
                    except Exception as e:
                        print(f"PatientID 확인 중 오류: {e}")
                
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
                    'patient_id': actual_patient_id,
                    'patient_name': patient_name,
                    'errors': errors if errors else None
                })
            except Exception as e:
                return Response({
                    'success': False,
                    'error': f'NIfTI 파일 처리 중 오류: {str(e)}',
                    'traceback': traceback.format_exc()
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # DICOM 파일인 경우
            dicom_data = uploaded_file.read()
            
            # patient_id가 제공된 경우 DICOM 파일의 PatientID 태그 수정
            if patient_id:
                try:
                    import pydicom
                    from io import BytesIO
                    
                    # DICOM 파일 읽기
                    dicom_file = pydicom.dcmread(BytesIO(dicom_data))
                    
                    # 한글 지원을 위해 문자셋 설정
                    dicom_file.SpecificCharacterSet = 'ISO_IR 192'  # UTF-8
                    
                    # PatientID와 PatientName 수정
                    dicom_file.PatientID = str(patient_id)
                    dicom_file.PatientName = str(patient_name)  # DB에서 가져온 실제 이름 사용
                    
                    print(f"Modifying DICOM tags: ID={patient_id}, Name={patient_name}")
                    
                    # 수정된 DICOM을 바이트로 변환
                    output = BytesIO()
                    pydicom.dcmwrite(output, dicom_file, write_like_original=False)
                    dicom_data = output.getvalue()
                except Exception as e:
                    print(f"DICOM 파일 태그 수정 실패 (원본 파일 그대로 업로드): {e}")
            
            # Orthanc에 업로드
            result = client.upload_dicom(dicom_data)
            
            # 업로드된 인스턴스의 Patient ID 확인
            actual_patient_id = patient_id or "UNKNOWN"
            try:
                if 'ID' in result:
                    instance_id = result['ID']
                    instance_info = client.get_instance_info(instance_id)
                    tags = instance_info.get('MainDicomTags', {})
                    if 'PatientID' in tags:
                        actual_patient_id = tags['PatientID']
            except Exception as e:
                print(f"Patient ID 확인 중 오류 (무시): {e}")
            
            return Response({
                'success': True,
                'result': result,
                'patient_id': actual_patient_id,
                'patient_name': patient_name,
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


@api_view(['GET'])
def orthanc_segmentation(request, patient_id):
    """
    환자의 세그멘테이션 데이터 조회
    향후 AI 모델 연동 시 실제 세그멘테이션 결과 반환
    """
    try:
        # TODO: 실제 AI 모델 세그멘테이션 로직 추가
        # 현재는 세그멘테이션 준비 상태만 반환
        client = OrthancClient()
        patient = client.find_patient_by_patient_id(patient_id)
        
        if not patient:
            return Response({
                'success': False,
                'error': 'Patient not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # 세그멘테이션 데이터 준비 중 (추후 실제 데이터로 대체)
        return Response({
            'success': True,
            'patient_id': patient_id,
            'segmentation_available': False,  # 실제 모델 연동 후 True로 변경
            'message': 'AI 세그멘테이션 모델 준비 중입니다.',
            'segmentation_data': None  # 실제 세그멘테이션 결과가 들어갈 위치
        })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def orthanc_run_segmentation(request, patient_id):
    """
    환자의 DICOM 이미지에 대해 AI 세그멘테이션 실행
    향후 실제 AI 모델 통합 예정
    """
    try:
        client = OrthancClient()
        patient = client.find_patient_by_patient_id(patient_id)
        
        if not patient:
            return Response({
                'success': False,
                'error': 'Patient not found'
            }, status=status.HTTP_404_NOT_FOUND)
        
        # TODO: 실제 AI 모델 세그멘테이션 실행 로직
        # 1. Orthanc에서 DICOM 이미지 가져오기
        # 2. AI 모델에 전달하여 세그멘테이션 실행
        # 3. 결과를 Orthanc에 저장 또는 별도 저장소에 저장
        # 4. 세그멘테이션 결과 메타데이터 반환
        
        import time
        time.sleep(2)  # 시뮬레이션 지연
        
        return Response({
            'success': True,
            'patient_id': patient_id,
            'segmentation_complete': True,
            'message': 'AI 세그멘테이션이 완료되었습니다. (시뮬레이션)',
            'result': {
                'tumor_detected': True,
                'tumor_volume_mm3': 1234.56,
                'confidence': 0.92
            }
        })
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

