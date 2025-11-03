# Flask ML Service 배포 가이드

## Flask 서비스 실행 방법

### Production 환경 (GCP VM)

Flask 서비스는 **systemd**를 통해 자동으로 실행됩니다.

#### 1. 서비스 파일 위치
```bash
/etc/systemd/system/ml-service.service
```

#### 2. 서비스 설정 내용
```ini
[Unit]
Description=ML Service (Flask) for Lung Cancer Prediction
After=network.target

[Service]
User=shrjsdn908
Group=shrjsdn908
WorkingDirectory=/srv/django-react/app/backend
Environment=PATH=/srv/django-react/app/backend/.venv/bin
ExecStart=/srv/django-react/app/backend/.venv/bin/gunicorn --workers 2 --bind 127.0.0.1:5002 ml_service.app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 3. 서비스 관리 명령어

**서비스 상태 확인:**
```bash
sudo systemctl status ml-service
```

**서비스 시작:**
```bash
sudo systemctl start ml-service
```

**서비스 재시작:**
```bash
sudo systemctl restart ml-service
```

**서비스 중지:**
```bash
sudo systemctl stop ml-service
```

**서비스 로그 확인:**
```bash
sudo journalctl -u ml-service -f
```

### Development 환경 (로컬)

로컬 개발 환경에서는 직접 Flask 앱을 실행할 수 있습니다:

```bash
cd backend
source .venv/bin/activate
python ml_service/app.py
```

개발 모드로 실행되면 `http://localhost:5002`에서 접근 가능합니다.

## 서비스 구조

### 1. Flask 앱 파일
- **위치**: `backend/ml_service/app.py`
- **엔드포인트**:
  - `GET /health` - 헬스 체크
  - `POST /predict` - 폐암 예측

### 2. ML 모델 파일
- **lung_cancer_model.pkl**: 학습된 폐암 예측 모델
- **feature_names.pkl**: 특성 이름 리스트
- **위치**: `backend/lung_cancer/ml_model/`

### 3. 모델 로드 과정
```python
# ml_service/app.py
model_path = os.path.join(current_dir, '..', 'lung_cancer', 'ml_model', 'lung_cancer_model.pkl')
model = joblib.load(model_path)
feature_names = joblib.load(feature_path)
```

## Django-Flask 통신

### 1. Django에서 Flask 호출
```python
# lung_cancer/views.py
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'http://localhost:5002')

ml_response = requests.post(
    f'{ML_SERVICE_URL}/predict',
    json={
        'gender': 'M',
        'age': 45,
        'smoking': True,
        # ... 기타 증상
    },
    timeout=10
)
```

### 2. Flask 응답 처리
```python
ml_result = ml_response.json()
# {
#     'prediction': 'YES',
#     'probability': 85.2,
#     'risk_level': '높음',
#     'risk_message': '...',
#     'symptoms': {...}
# }
```

## 배포 프로세스

### 자동 배포 (GitHub Actions)

배포 시 자동으로 다음 작업이 수행됩니다:

1. **코드 동기화**: RSYNC로 GCP VM에 코드 전송
2. **Python 패키지 설치**: `pip install -r requirements.txt`
3. **서비스 파일 생성**: systemd 서비스 파일 생성
4. **서비스 재시작**: `systemctl daemon-reload && systemctl restart ml-service`

### 수동 배포

```bash
# 1. 코드 동기화 후
cd /srv/django-react/app/backend

# 2. Python 환경 활성화
source .venv/bin/activate

# 3. 의존성 확인
pip install -r requirements.txt

# 4. 서비스 재시작
sudo systemctl restart ml-service

# 5. 상태 확인
sudo systemctl status ml-service
```

## 포트 및 접근성

- **내부 포트**: `127.0.0.1:5002` (localhost만 접근 가능)
- **외부 접근**: 불가 (Django를 통해서만 호출)
- **보안**: Flask 서비스는 직접 외부에서 접근할 수 없도록 설정됨

## 트러블슈팅

### 1. 서비스가 시작되지 않는 경우
```bash
# 로그 확인
sudo journalctl -u ml-service -n 50

# 모델 파일 확인
ls -la /srv/django-react/app/backend/lung_cancer/ml_model/

# Python 환경 확인
cd /srv/django-react/app/backend
source .venv/bin/activate
python -c "import joblib; print('OK')"
```

### 2. 예측이 실패하는 경우
```bash
# Flask 서비스 직접 테스트
curl -X GET http://localhost:5002/health

# 예측 테스트
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"M","age":45,"smoking":true}'
```

### 3. 모델 로드 실패
- 모델 파일 경로 확인: `backend/lung_cancer/ml_model/`
- Python 패키지 확인: `pip install scikit-learn joblib pandas numpy`
- 파일 권한 확인: `chmod 644 *.pkl`

## 성능 최적화

- **Workers**: 2개 (현재 설정)
- **Timeout**: 10초 (Django 측 설정)
- **Model Loading**: 앱 시작 시 한 번만 로드
- **Cache**: 모델은 메모리에 상주

## 모니터링

### 헬스 체크
```bash
curl http://localhost:5002/health
# 응답: {"status":"healthy","model_loaded":true}
```

### 로그 모니터링
```bash
# 실시간 로그
sudo journalctl -u ml-service -f

# 최근 100줄
sudo journalctl -u ml-service -n 100
```

## 참고 자료

- Flask: https://flask.palletsprojects.com/
- Gunicorn: https://gunicorn.org/
- scikit-learn: https://scikit-learn.org/
- joblib: https://joblib.readthedocs.io/
