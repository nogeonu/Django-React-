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
        self.username = username or os.getenv('ORTHANC_USER')
        self.password = password or os.getenv('ORTHANC_PASSWORD')
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
        """환자 ID 목록"""
        response = requests.get(f"{self.base_url}/patients", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_patient_info(self, patient_id: str) -> Dict[str, Any]:
        """환자 상세 정보"""
        response = requests.get(f"{self.base_url}/patients/{patient_id}", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_patient_studies(self, patient_id: str) -> List[str]:
        """환자의 Study 목록"""
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

