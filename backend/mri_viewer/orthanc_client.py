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
        try:
            # Orthanc의 find API 사용
            query = {"PatientID": patient_id}
            response = requests.post(
                f"{self.base_url}/tools/find",
                json=query,
                auth=self.auth
            )
            response.raise_for_status()
            patient_ids = response.json()
            
            # 반환된 ID 중 첫 번째 사용 (일반적으로 하나)
            if patient_ids and len(patient_ids) > 0:
                return patient_ids[0]
            
            # find가 실패하면 모든 환자를 순회하면서 PatientID 태그 확인
            all_patients = self.get_patients()
            for orthanc_patient_id in all_patients:
                try:
                    info = self.get_patient_info(orthanc_patient_id)
                    tags = info.get('MainDicomTags', {})
                    if tags.get('PatientID') == patient_id:
                        return orthanc_patient_id
                except:
                    continue
            
            return None
        except Exception as e:
            print(f"Error finding patient by PatientID {patient_id}: {e}")
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

