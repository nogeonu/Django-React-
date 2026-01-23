# FastAPI 약물 검색 서비스 설정 가이드

FastAPI 서버(`http://34.42.223.43:8002`)가 실행되지 않아 약물 검색이 실패하는 경우, 다음 단계를 따라 설정하세요.

## 📋 사전 준비

1. **CDSS_Final_Package 파일 확인**
   - `CDSS_Final_Package/backend/` 폴더에 다음 파일들이 있어야 합니다:
     - `main.py`
     - `requirements.txt`
     - `ai_service.py`
     - `ddinter_helper.py`
     - `drug_dictionary.py`
     - `ingredient_translator.py`
     - `.env` (데이터베이스 접속 정보 포함)

## 🚀 서버에 배포하기

### 방법 1: 수동 배포 (권장)

1. **SSH로 서버 접속**
   ```bash
   ssh shrjsdn908@34.42.223.43
   ```

2. **디렉토리 생성 및 파일 복사**
   ```bash
   # 서버에서 실행
   APP_DIR="/srv/django-react/app"
   mkdir -p "$APP_DIR/backend/drug_api"
   
   # 로컬에서 실행 (Mac)
   cd /Users/nogeon-u/Desktop/건양대_바이오메디컬/Django
   scp -r CDSS_Final_Package/backend/* shrjsdn908@34.42.223.43:/srv/django-react/app/backend/drug_api/
   ```

3. **의존성 설치**
   ```bash
   # 서버에서 실행
   cd /srv/django-react/app/backend
   source .venv/bin/activate
   pip install fastapi uvicorn pymysql python-dotenv openai requests pydantic
   ```

4. **systemd 서비스 설정**
   ```bash
   # 서버에서 실행
   sudo bash /srv/django-react/app/scripts/setup_drug_api_service.sh
   ```

5. **서비스 시작 및 확인**
   ```bash
   # 서비스 상태 확인
   sudo systemctl status drug-api-service
   
   # 서비스 시작
   sudo systemctl start drug-api-service
   
   # 로그 확인
   sudo journalctl -u drug-api-service -f
   ```

### 방법 2: 자동 배포 (GitHub Actions)

배포 워크플로우에 FastAPI 서비스 설정이 포함되어 있습니다. 하지만 `CDSS_Final_Package/backend` 파일들을 서버에 수동으로 복사해야 합니다.

## ✅ 확인 방법

1. **서비스 상태 확인**
   ```bash
   sudo systemctl status drug-api-service
   ```

2. **포트 확인**
   ```bash
   sudo netstat -tlnp | grep 8002
   # 또는
   curl http://localhost:8002/docs
   ```

3. **브라우저에서 확인**
   - `http://34.42.223.43:8002/docs` 접속
   - FastAPI Swagger UI가 표시되면 성공

## 🔧 문제 해결

### 서비스가 시작되지 않는 경우

1. **로그 확인**
   ```bash
   sudo journalctl -u drug-api-service -n 50
   ```

2. **수동 실행 테스트**
   ```bash
   cd /srv/django-react/app/backend/drug_api
   source ../.venv/bin/activate
   uvicorn main:app --host 0.0.0.0 --port 8002
   ```

3. **데이터베이스 연결 확인**
   - `.env` 파일의 DB 접속 정보 확인
   - MySQL 서비스 실행 확인: `sudo systemctl status mysql`

### 포트가 열리지 않는 경우

1. **방화벽 확인**
   ```bash
   sudo ufw status
   sudo ufw allow 8002/tcp
   ```

2. **GCP 방화벽 규칙 확인**
   - GCP 콘솔 → VPC 네트워크 → 방화벽 규칙
   - 포트 8002 인바운드 허용 규칙 추가

## 📝 참고

- FastAPI 서비스는 `http://0.0.0.0:8002`에서 실행됩니다
- 서비스는 자동으로 재시작되도록 설정되어 있습니다
- 로그는 `journalctl -u drug-api-service`로 확인할 수 있습니다
