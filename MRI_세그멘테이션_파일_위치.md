# MRI 세그멘테이션 파일 위치 정리

## 📁 현재 프로젝트 내 파일 위치

### 1. Django 백엔드
- **`backend/mri_viewer/segmentation_views.py`**
  - Django REST API 엔드포인트
  - Mosec 서비스와 통신
  - Orthanc과 통신하여 DICOM 파일 처리

### 2. 서버에 있는 Mosec 서비스 파일
- **`/home/shrjsdn908/segmentation_mosec.py`** (서버에 직접 위치)
  - Mosec 워커 서비스
  - SwinUNETR 모델 로드 및 추론
  - DICOM SEG 파일 생성

### 3. 프론트엔드
- **`frontend/src/pages/MRIViewer.tsx`**
  - MRI 뷰어 페이지
  - 세그멘테이션 실행 버튼 및 결과 표시

- **`frontend/src/pages/MRIImageDetail.tsx`**
  - MRI 이미지 상세 페이지
  - 세그멘테이션 오버레이 표시

- **`frontend/src/components/CornerstoneViewer.tsx`**
  - DICOM 뷰어 컴포넌트
  - 세그멘테이션 오버레이 렌더링

## 🔄 조원 코드 교체 시 업데이트할 파일

### 백엔드 (Django)
1. `backend/mri_viewer/segmentation_views.py` - API 엔드포인트
2. 서버의 `/home/shrjsdn908/segmentation_mosec.py` - Mosec 서비스

### 프론트엔드 (React)
1. `frontend/src/pages/MRIViewer.tsx` - 세그멘테이션 실행 UI
2. `frontend/src/pages/MRIImageDetail.tsx` - 세그멘테이션 결과 표시
3. `frontend/src/components/CornerstoneViewer.tsx` - 오버레이 렌더링

## 📝 참고사항

- Mosec 서비스는 서버에서 직접 실행되므로 서버에 접속하여 파일을 교체해야 함
- 서비스 재시작: `sudo systemctl restart dl-service.service`
- 로그 확인: `journalctl -u dl-service.service -f`
