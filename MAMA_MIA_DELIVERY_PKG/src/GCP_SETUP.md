# GCP 환경 설정 가이드

## 중요: pydicom 버전 호환성

### ⚠️ 필수 확인 사항
```bash
# pydicom-seg는 pydicom 2.x만 지원
# pydicom 3.x 설치 시 에러 발생:
# ModuleNotFoundError: No module named 'pydicom._storage_sopclass_uids'
```

### 올바른 설치 순서
```bash
# 1. 기존 pydicom 제거 (3.x 버전인 경우)
pip uninstall pydicom -y

# 2. pydicom 2.4.4 설치
pip install pydicom==2.4.4

# 3. pydicom-seg 설치
pip install pydicom-seg==0.4.1

# 4. 추가 DICOM 라이브러리
pip install highdicom SimpleITK

# 5. 확인
python3 -c "import pydicom; print(f'pydicom: {pydicom.__version__}')"
python3 -c "import pydicom_seg; print('pydicom-seg: OK')"
```

---

## GCP VM 인스턴스 설정

### 시스템 패키지
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0
```

### Python 환경
```bash
# 가상환경 생성 (권장)
python3 -m venv venv
source venv/bin/activate

# requirements.txt 설치
pip install -r requirements.txt
```

---

## 서비스 설정 (systemd)

### 서비스 파일 생성
```bash
sudo nano /etc/systemd/system/segmentation-service.service
```

### 서비스 내용
```ini
[Unit]
Description=Phase 1 Segmentation Service
After=network.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/Phase1_Segmentation
Environment="PATH=/home/YOUR_USER/venv/bin"
ExecStart=/home/YOUR_USER/venv/bin/python api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 서비스 시작
```bash
sudo systemctl daemon-reload
sudo systemctl enable segmentation-service
sudo systemctl start segmentation-service
sudo systemctl status segmentation-service
```

---

## 문제 해결

### pydicom 버전 충돌
```bash
# 증상: ModuleNotFoundError: pydicom._storage_sopclass_uids
# 해결: pydicom 다운그레이드
pip install --force-reinstall pydicom==2.4.4
```

### 메모리 부족
```bash
# Cloud Run 메모리 증가
gcloud run deploy --memory 8Gi
```

### GPU 필요 시
```bash
# Cloud Run은 GPU 미지원
# GKE 사용 필요
```

---

## 환경 변수

### .env 파일 (선택)
```bash
MODEL_PATH=/path/to/best_model.pth
PORT=8080
LOG_LEVEL=INFO
```

---

## 모니터링

### 로그 확인
```bash
# systemd 서비스
sudo journalctl -u segmentation-service -f

# Cloud Run
gcloud run logs read segmentation-service --limit 50
```

### Health Check
```bash
curl http://localhost:8080/health
```
