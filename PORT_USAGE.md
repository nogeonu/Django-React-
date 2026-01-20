# 포트 사용 현황

서버에서 사용 중인 포트 목록입니다.

## 현재 사용 중인 포트

| 포트 | 서비스 | 설명 | 외부 접근 | 상태 |
|------|--------|------|-----------|------|
| **80** | Nginx | 웹 서버 (프론트엔드 + API 프록시) | ✅ 공개 | ✅ 실행 중 |
| **8000** | Django (Gunicorn) | 메인 백엔드 API | ❌ 내부만 (Nginx 프록시) | ✅ 실행 중 |
| **8001** | 챗봇 서버 (Gunicorn) | Flutter 프로젝트 챗봇 서버 (별도 Django 프로젝트) | ✅ 공개 | ✅ 실행 중 |
| | | ⚠️ **같은 데이터베이스 공유**: `hospital_db` (MySQL) | | |
| | | ⚠️ **예약 연동**: Flutter 챗봇에서 예약 생성 시 Django 프로젝트에도 반영 | | |
| **8002** | FastAPI | 약물 검색 및 상호작용 검사 | ❌ 내부만 (Nginx 프록시) | ✅ 실행 중 |
| **5002** | Flask ML Service | 폐암 ML 서비스 + pCR 예측 (유전자 예측) | ❌ 내부만 | ✅ 실행 중 |
| **5006** | Mosec | MRI 세그멘테이션 | ❌ 내부만 | ✅ 실행 중 |
| **5007** | Mosec | 맘모그래피 분석 | ❌ 내부만 | ✅ 실행 중 |
| **5008** | Mosec | 병리 이미지 분석 | ❌ 내부만 | ✅ 실행 중 |
| **8042** | Orthanc | DICOM 서버 | ❌ 내부만 (Nginx 프록시) | ✅ 실행 중 |

## 접근 경로

### 외부에서 접근 가능한 경로 (포트 80)

- `http://34.42.223.43/` - 프론트엔드
- `http://34.42.223.43/api/` - Django API
- `http://34.42.223.43/drug-api/` - FastAPI 약물 검색 서비스
- `http://34.42.223.43/orthanc/` - Orthanc DICOM 서버

### 내부에서만 접근 가능

- `http://127.0.0.1:8000` - Django (Gunicorn)
- `http://127.0.0.1:8002` - FastAPI 약물 검색
- `http://127.0.0.1:5002` - Flask ML Service
- `http://127.0.0.1:5006` - Mosec (MRI)
- `http://127.0.0.1:5007` - Mosec (맘모그래피)
- `http://127.0.0.1:5008` - Mosec (병리)
- `http://127.0.0.1:8042` - Orthanc

### 별도 프로젝트 (Flutter 챗봇)

- `http://34.42.223.43:8001` - Flutter 프로젝트 챗봇 서버 (별도 Django 프로젝트)
  - 위치: `/Flutter/flutter_app/chatbot_server/`
  - Django 프로젝트(`Django-React--main`)의 `chatbot` 앱과는 별개
  - Django 프로젝트의 챗봇은 8000 포트의 `/api/chat/` 엔드포인트 사용
  - **데이터베이스 연동**: 같은 `hospital_db` (MySQL) 사용
  - **예약 연동**: Flutter 챗봇에서 예약 생성 시 `patients_appointment` 테이블에 직접 INSERT하여 Django 프로젝트에도 즉시 반영됨

## 서비스 관리

### 서비스 상태 확인
```bash
sudo systemctl status <service-name>
```

### 주요 서비스 목록
- `nginx.service` - 웹 서버
- `gunicorn.service` - Django 백엔드
- `drug-api-service.service` - FastAPI 약물 검색
- `ml-service.service` - Flask ML Service (폐암 + pCR 예측)
- `segmentation-mosec.service` - MRI 세그멘테이션 (포트 5006)
- `mammography-mosec.service` - 맘모그래피 (포트 5007)
- `pathology-mosec.service` - 병리 이미지 (포트 5008)

## 포트 충돌 확인

새로운 서비스를 추가할 때 포트 충돌을 확인:
```bash
sudo ss -tlnp | grep <포트번호>
```

## 참고

- 대부분의 서비스는 내부(127.0.0.1)에서만 리스닝하며, Nginx를 통해 외부에 노출됩니다.
- 포트 80만 외부에서 직접 접근 가능하며, 나머지는 Nginx 프록시를 통해 접근합니다.
- GCP 방화벽 규칙에서 포트 80만 열려 있으면 됩니다.
