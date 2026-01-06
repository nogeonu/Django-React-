"""
Orthanc PACS 서버 REST API 클라이언트
"""
import requests
from typing import Optional, List, Dict, Any
import os


class OrthancClient:
    """Orthanc REST API 클라이언트"""
    
    def __init__(
        self, 
        base_url: str = None, 
        username: str = None, 
        password: str = None
    ):
        self.base_url = (base_url or os.getenv('ORTHANC_URL', 'http://localhost:8042')).rstrip('/')
        self.username = username or os.getenv('ORTHANC_USER', 'admin')
        self.password = password or os.getenv('ORTHANC_PASSWORD', 'admin123')
        self.auth = (self.username, self.password) if self.username and self.password else None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Orthanc 시스템 정보"""
        response = requests.get(f"{self.base_url}/system", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Orthanc 통계 정보"""
        response = requests.get(f"{self.base_url}/statistics", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_patients(self) -> List[str]:
        """환자 ID 목록 (Orthanc 내부 ID)"""
        response = requests.get(f"{self.base_url}/patients", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def find_patient_by_patient_id(self, patient_id: str) -> Optional[str]:
        """DICOM PatientID 태그로 환자 찾기 (Orthanc 내부 ID 반환)"""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.debug(f"Searching for PatientID: '{patient_id}'")
            
            # 방법 1: 간단한 형식 시도
            query1 = {"PatientID": patient_id}
            logger.debug(f"Trying Orthanc /tools/find with simple format: {query1}")
            try:
                response = requests.post(
                    f"{self.base_url}/tools/find",
                    json=query1,
                    auth=self.auth
                )
                response.raise_for_status()
                patient_ids = response.json()
                logger.debug(f"/tools/find (simple) returned {len(patient_ids) if patient_ids else 0} results: {patient_ids}")
                
                if patient_ids and len(patient_ids) > 0:
                    logger.info(f"Found patient via /tools/find (simple): {patient_ids[0]} for PatientID '{patient_id}'")
                    return patient_ids[0]
            except Exception as e1:
                logger.debug(f"Simple format failed: {e1}")
            
            # 방법 2: 구조화된 형식 시도
            query2 = {"Level": "Patient", "Query": {"PatientID": patient_id}}
            logger.debug(f"Trying Orthanc /tools/find with structured format: {query2}")
            try:
                response = requests.post(
                    f"{self.base_url}/tools/find",
                    json=query2,
                    auth=self.auth
                )
                response.raise_for_status()
                patient_ids = response.json()
                logger.debug(f"/tools/find (structured) returned {len(patient_ids) if patient_ids else 0} results: {patient_ids}")
                
                if patient_ids and len(patient_ids) > 0:
                    logger.info(f"Found patient via /tools/find (structured): {patient_ids[0]} for PatientID '{patient_id}'")
                    return patient_ids[0]
            except Exception as e2:
                logger.debug(f"Structured format failed: {e2}")
            
            # find가 실패하면 모든 환자를 순회하면서 PatientID 태그 확인
            logger.warning(f"/tools/find did not find PatientID '{patient_id}', falling back to iteration")
            all_patients = self.get_patients()
            logger.debug(f"Found {len(all_patients)} total patients in Orthanc, iterating...")
            
            found_patient_ids = []  # 디버깅용: 실제 저장된 PatientID 목록
            for orthanc_patient_id in all_patients:
                try:
                    # 직접 API 호출 (get_patient_info 호출하지 않음)
                    response = requests.get(
                        f"{self.base_url}/patients/{orthanc_patient_id}",
                        auth=self.auth
                    )
                    response.raise_for_status()
                    info = response.json()
                    tags = info.get('MainDicomTags', {})
                    stored_patient_id = tags.get('PatientID', '')
                    
                    # 디버깅: 처음 몇 개만 로깅
                    if len(found_patient_ids) < 10:
                        found_patient_ids.append((orthanc_patient_id, stored_patient_id))
                    
                    # 정확히 일치하는 경우 먼저 확인 (가장 확실)
                    if stored_patient_id == patient_id:
                        logger.info(f"Found patient via iteration (exact match): {orthanc_patient_id} for PatientID '{patient_id}'")
                        return orthanc_patient_id
                    # 대소문자 구분 없이 비교 (PatientID는 보통 대소문자 구분 없음)
                    elif stored_patient_id and stored_patient_id.strip().upper() == patient_id.strip().upper():
                        logger.info(f"Found patient via iteration (case-insensitive): {orthanc_patient_id} for PatientID '{patient_id}' (stored as '{stored_patient_id}')")
                        return orthanc_patient_id
                except Exception as e:
                    logger.debug(f"Error checking patient {orthanc_patient_id}: {e}")
                    continue
            
            # 로깅: 찾지 못했을 때 실제 저장된 PatientID 목록 출력
            logger.warning(f"PatientID '{patient_id}' not found. Sample stored PatientIDs: {found_patient_ids}")
            return None
        except Exception as e:
            logger.error(f"Error finding patient by PatientID {patient_id}: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            return None
    
    def get_patient_info(self, patient_id: str) -> Dict[str, Any]:
        """환자 상세 정보 (Orthanc 내부 ID 또는 DICOM PatientID 사용 가능)"""
        # 먼저 직접 접근 시도 (Orthanc 내부 ID인 경우)
        try:
            response = requests.get(f"{self.base_url}/patients/{patient_id}", auth=self.auth)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Orthanc 내부 ID가 아니면 DICOM PatientID로 검색
                found_id = self.find_patient_by_patient_id(patient_id)
                if found_id:
                    response = requests.get(f"{self.base_url}/patients/{found_id}", auth=self.auth)
                    response.raise_for_status()
                    return response.json()
            raise
    
    def get_patient_studies(self, patient_id: str) -> List[str]:
        """환자의 Study 목록 (Orthanc 내부 ID 사용)"""
        response = requests.get(f"{self.base_url}/patients/{patient_id}/studies", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_study_info(self, study_id: str) -> Dict[str, Any]:
        """Study 상세 정보"""
        response = requests.get(f"{self.base_url}/studies/{study_id}", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_existing_study_instance_uid(self, patient_id: str) -> Optional[str]:
        """
        환자의 기존 Study에서 StudyInstanceUID 반환
        같은 환자는 하나의 Study로 통합하기 위함
        
        Returns:
            StudyInstanceUID (str) 또는 None (기존 Study가 없는 경우)
        """
        try:
            # DICOM PatientID로 환자 찾기
            orthanc_patient_id = self.find_patient_by_patient_id(patient_id)
            if not orthanc_patient_id:
                return None
            
            # 환자의 Study 목록 가져오기
            studies = self.get_patient_studies(orthanc_patient_id)
            if not studies or len(studies) == 0:
                return None
            
            # 첫 번째 Study의 StudyInstanceUID 반환
            # (같은 환자의 모든 영상은 하나의 Study로 통합)
            study_id = studies[0] if isinstance(studies[0], str) else studies[0].get('ID')
            study_info = self.get_study_info(study_id)
            
            # MainDicomTags에서 StudyInstanceUID 추출
            tags = study_info.get('MainDicomTags', {})
            study_instance_uid = tags.get('StudyInstanceUID')
            
            return study_instance_uid
        except Exception as e:
            logger.warning(f"Failed to get existing StudyInstanceUID for patient {patient_id}: {e}")
            return None
    
    def get_study_series(self, study_id: str) -> List[str]:
        """Study의 Series 목록"""
        response = requests.get(f"{self.base_url}/studies/{study_id}/series", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """Series 상세 정보"""
        response = requests.get(f"{self.base_url}/series/{series_id}", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_series_instances(self, series_id: str) -> List[str]:
        """Series의 Instance 목록"""
        response = requests.get(f"{self.base_url}/series/{series_id}/instances", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_instance_info(self, instance_id: str) -> Dict[str, Any]:
        """Instance 상세 정보"""
        response = requests.get(f"{self.base_url}/instances/{instance_id}", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_instance_preview(self, instance_id: str) -> bytes:
        """Instance 미리보기 이미지 (PNG)"""
        response = requests.get(f"{self.base_url}/instances/{instance_id}/preview", auth=self.auth)
        response.raise_for_status()
        return response.content
    
    def get_instance_file(self, instance_id: str) -> bytes:
        """Instance DICOM 파일"""
        response = requests.get(f"{self.base_url}/instances/{instance_id}/file", auth=self.auth)
        response.raise_for_status()
        return response.content
    
    def upload_dicom(self, dicom_data: bytes) -> Dict[str, Any]:
        """DICOM 파일 업로드"""
        response = requests.post(
            f"{self.base_url}/instances",
            data=dicom_data,
            headers={'Content-Type': 'application/dicom'},
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
    
    def delete_patient(self, patient_id: str) -> bool:
        """환자 데이터 삭제"""
        response = requests.delete(f"{self.base_url}/patients/{patient_id}", auth=self.auth)
        response.raise_for_status()
        return True
    
    def delete_study(self, study_id: str) -> bool:
        """Study 삭제"""
        response = requests.delete(f"{self.base_url}/studies/{study_id}", auth=self.auth)
        response.raise_for_status()
        return True

