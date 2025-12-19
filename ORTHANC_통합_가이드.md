# 🏥 Orthanc PACS 통합 가이드

## 📋 현재 상태

### Orthanc 서버 정보
- **URL**: http://34.42.223.43:8042
- **Web UI**: http://34.42.223.43/orthanc/ui/app/#/
- **DICOM 포트**: 4242
- **컨테이너**: orthancteam/orthanc

### Orthanc REST API 엔드포인트

Orthanc는 기본적으로 다음 REST API를 제공합니다:

```
GET  /system              - 시스템 정보
GET  /patients            - 환자 목록
GET  /patients/{id}       - 환자 상세 정보
GET  /studies             - Study 목록
GET  /series              - Series 목록
GET  /instances           - Instance 목록
POST /instances           - DICOM 파일 업로드
GET  /instances/{id}/file - DICOM 파일 다운로드
GET  /instances/{id}/preview - 이미지 미리보기 (PNG)
```

## 🔧 Django에서 Orthanc API 사용

### 1. Orthanc 클라이언트 생성

```python
# backend/mri_viewer/orthanc_client.py
import requests
from typing import Optional, List, Dict

class OrthancClient:
    def __init__(self, base_url: str = "http://localhost:8042", username: str = None, password: str = None):
        self.base_url = base_url.rstrip('/')
        self.auth = (username, password) if username and password else None
    
    def get_system_info(self) -> Dict:
        """Orthanc 시스템 정보"""
        response = requests.get(f"{self.base_url}/system", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_patients(self) -> List[str]:
        """환자 ID 목록"""
        response = requests.get(f"{self.base_url}/patients", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_patient_info(self, patient_id: str) -> Dict:
        """환자 상세 정보"""
        response = requests.get(f"{self.base_url}/patients/{patient_id}", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_patient_studies(self, patient_id: str) -> List[str]:
        """환자의 Study 목록"""
        response = requests.get(f"{self.base_url}/patients/{patient_id}/studies", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_study_series(self, study_id: str) -> List[str]:
        """Study의 Series 목록"""
        response = requests.get(f"{self.base_url}/studies/{study_id}/series", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_series_instances(self, series_id: str) -> List[str]:
        """Series의 Instance 목록"""
        response = requests.get(f"{self.base_url}/series/{series_id}/instances", auth=self.auth)
        response.raise_for_status()
        return response.json()
    
    def get_instance_preview(self, instance_id: str) -> bytes:
        """Instance 미리보기 이미지 (PNG)"""
        response = requests.get(f"{self.base_url}/instances/{instance_id}/preview", auth=self.auth)
        response.raise_for_status()
        return response.content
    
    def upload_dicom(self, dicom_data: bytes) -> Dict:
        """DICOM 파일 업로드"""
        response = requests.post(
            f"{self.base_url}/instances",
            data=dicom_data,
            headers={'Content-Type': 'application/dicom'},
            auth=self.auth
        )
        response.raise_for_status()
        return response.json()
```

### 2. Django View 추가

```python
# backend/mri_viewer/orthanc_views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse
from .orthanc_client import OrthancClient
import os

ORTHANC_URL = os.getenv('ORTHANC_URL', 'http://localhost:8042')
ORTHANC_USER = os.getenv('ORTHANC_USER')
ORTHANC_PASSWORD = os.getenv('ORTHANC_PASSWORD')

@api_view(['GET'])
def orthanc_patients(request):
    """Orthanc 환자 목록"""
    client = OrthancClient(ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD)
    try:
        patient_ids = client.get_patients()
        patients = []
        for patient_id in patient_ids:
            info = client.get_patient_info(patient_id)
            patients.append({
                'id': patient_id,
                'info': info
            })
        return Response({'success': True, 'patients': patients})
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)

@api_view(['GET'])
def orthanc_patient_images(request, patient_id):
    """환자의 이미지 목록"""
    client = OrthancClient(ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD)
    try:
        studies = client.get_patient_studies(patient_id)
        images = []
        
        for study_id in studies:
            series_list = client.get_study_series(study_id)
            for series_id in series_list:
                instances = client.get_series_instances(series_id)
                for instance_id in instances:
                    images.append({
                        'study_id': study_id,
                        'series_id': series_id,
                        'instance_id': instance_id,
                        'preview_url': f'/api/mri/orthanc/instances/{instance_id}/preview/'
                    })
        
        return Response({'success': True, 'images': images})
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)

@api_view(['GET'])
def orthanc_instance_preview(request, instance_id):
    """이미지 미리보기"""
    client = OrthancClient(ORTHANC_URL, ORTHANC_USER, ORTHANC_PASSWORD)
    try:
        image_data = client.get_instance_preview(instance_id)
        return HttpResponse(image_data, content_type='image/png')
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)
```

### 3. URL 설정

```python
# backend/mri_viewer/urls.py
from django.urls import path
from . import views, orthanc_views

urlpatterns = [
    # 기존 MRI Viewer API
    path('patients/', views.get_patient_list, name='mri-patient-list'),
    path('patients/<str:patient_id>/', views.get_patient_info, name='mri-patient-info'),
    path('patients/<str:patient_id>/slice/', views.get_mri_slice, name='mri-slice'),
    path('patients/<str:patient_id>/volume/', views.get_volume_info, name='mri-volume-info'),
    
    # Orthanc API
    path('orthanc/patients/', orthanc_views.orthanc_patients, name='orthanc-patients'),
    path('orthanc/patients/<str:patient_id>/images/', orthanc_views.orthanc_patient_images, name='orthanc-patient-images'),
    path('orthanc/instances/<str:instance_id>/preview/', orthanc_views.orthanc_instance_preview, name='orthanc-instance-preview'),
]
```

## 🎨 프론트엔드에서 Orthanc 이미지 표시

### React 컴포넌트

```typescript
// frontend/src/pages/MRIViewer.tsx에 추가

const [orthancImages, setOrthancImages] = useState<any[]>([]);

// Orthanc에서 환자 이미지 가져오기
const fetchOrthancImages = async (patientId: string) => {
  try {
    const response = await fetch(`/api/mri/orthanc/patients/${patientId}/images/`);
    const data = await response.json();
    if (data.success) {
      setOrthancImages(data.images);
    }
  } catch (error) {
    console.error('Orthanc 이미지 로드 실패:', error);
  }
};

// 이미지 표시
{orthancImages.map((image, index) => (
  <img
    key={index}
    src={image.preview_url}
    alt={`Instance ${image.instance_id}`}
    className="w-full h-auto"
  />
))}
```

## 🚀 배포 설정

### 환경 변수 설정

```bash
# backend/.env
ORTHANC_URL=http://localhost:8042
ORTHANC_USER=admin
ORTHANC_PASSWORD=your_password_here
```

### GCP VM에서 설정

```bash
# 1. Orthanc 컨테이너 확인
docker ps | grep orthanc

# 2. Orthanc 설정 확인
docker exec -it $(docker ps -q --filter "ancestor=orthancteam/orthanc") cat /etc/orthanc/orthanc.json

# 3. 환경 변수 설정
echo 'export ORTHANC_URL="http://localhost:8042"' >> ~/.bashrc
source ~/.bashrc
```

## 📊 Orthanc Web UI 접속

```
http://34.42.223.43/orthanc/ui/app/#/
```

여기서:
- 환자 목록 확인
- DICOM 이미지 업로드
- 이미지 미리보기
- Study/Series/Instance 관리

## 🔗 API 테스트

### curl로 테스트

```bash
# 시스템 정보
curl http://34.42.223.43:8042/system

# 환자 목록
curl http://34.42.223.43:8042/patients

# 환자 상세 정보
curl http://34.42.223.43:8042/patients/{patient_id}

# 이미지 미리보기
curl http://34.42.223.43:8042/instances/{instance_id}/preview -o preview.png
```

## 📝 다음 단계

1. ✅ Orthanc 서버 실행 확인
2. ⬜ Django에 Orthanc 클라이언트 추가
3. ⬜ API 엔드포인트 구현
4. ⬜ 프론트엔드에서 Orthanc 이미지 표시
5. ⬜ DICOM 업로드 기능 추가

이제 Orthanc REST API를 사용해서 의료 이미지를 관리할 수 있습니다!

