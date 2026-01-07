# GCP AI 서비스 수정 가이드

## 🚨 문제 진단

503 Service Unavailable 오류는 AI 서비스(YOLO 모델)가 실행되지 않아서 발생합니다.

### 발견된 문제점:
1. **포트 불일치**: AI 서비스는 5004 포트 사용 (코드에서 확인)
2. **모델 파일 경로**: `/home/shrjsdn908/models/yolo11_mammography/best.pt`
3. **Mosec 서비스 미실행**

---

## 📋 GCP 서버에서 실행할 명령어

### 1단계: SSH 접속
```bash
# GCP 콘솔에서 SSH 버튼 클릭 또는
gcloud compute ssh koyang-2510 --zone=<your-zone>
```

### 2단계: 진단 스크립트 실행
```bash
# 프로젝트 디렉토리로 이동
cd ~/Django-React-*

# 진단 스크립트 다운로드 및 실행
chmod +x diagnose_ai_service.sh
./diagnose_ai_service.sh
```

### 3단계: AI 서비스 수정 및 재시작
```bash
# 수정 스크립트 실행
chmod +x fix_ai_service.sh
./fix_ai_service.sh
```

### 4단계: Systemd 서비스 생성 (선택사항, 자동 재시작용)
```bash
chmod +x create_systemd_service.sh
./create_systemd_service.sh
```

---

## 🔧 수동 수정 방법

### 1. 프로세스 확인 및 종료
```bash
# 실행 중인 AI 서비스 확인
ps aux | grep -E "mammography|mosec|app.py"

# 종료
pkill -f "mammography.*app.py"
pkill -f "mosec"
```

### 2. 모델 파일 확인
```bash
# 모델 파일 찾기
find ~ -name "best.pt" -o -name "yolo*.pt"

# 모델 디렉토리 생성 (없는 경우)
mkdir -p ~/models/yolo11_mammography

# 모델 파일이 없으면 기본 YOLO 모델 다운로드
cd ~/Django-React-*/backend
source .venv/bin/activate
python3 -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.save('~/models/yolo11_mammography/best.pt')"
```

### 3. 필수 패키지 설치
```bash
cd ~/Django-React-*/backend
source .venv/bin/activate

pip install --upgrade pip
pip install ultralytics opencv-python torch torchvision mosec msgpack
```

### 4. AI 서비스 실행
```bash
cd ~/Django-React-*/backend/mammography_ai_service

# 백그라운드 실행
nohup python app.py > /tmp/mammography-ai.log 2>&1 &

# 로그 확인
tail -f /tmp/mammography-ai.log
```

### 5. 서비스 테스트
```bash
# 포트 확인
netstat -tlnp | grep 5004

# 헬스 체크
curl http://localhost:5004/health
```

---

## 🔍 문제 해결 체크리스트

- [ ] Python 가상환경 활성화 확인
- [ ] 필수 패키지 설치 확인 (ultralytics, mosec, torch)
- [ ] 모델 파일 존재 확인
- [ ] 포트 5004가 열려있는지 확인
- [ ] 방화벽 규칙 확인 (GCP 방화벽)
- [ ] AI 서비스 로그 확인
- [ ] Django 설정에서 AI 서비스 URL 확인

---

## 📝 로그 확인 방법

```bash
# AI 서비스 로그
tail -f /tmp/mammography-ai.log

# Systemd 서비스 로그 (서비스로 실행한 경우)
sudo journalctl -u mammography-ai.service -f

# Django 로그
tail -f ~/Django-React-*/backend/logs/django.log
```

---

## 🚀 빠른 해결 방법 (올인원)

```bash
# 1. 프로젝트 디렉토리로 이동
cd ~/Django-React-*

# 2. 기존 프로세스 종료
pkill -f "mammography.*app.py"

# 3. 가상환경 활성화 및 패키지 설치
cd backend
source .venv/bin/activate
pip install ultralytics opencv-python torch torchvision mosec msgpack -q

# 4. 모델 다운로드 (없는 경우)
python3 << EOF
from ultralytics import YOLO
import os
model_dir = os.path.expanduser('~/models/yolo11_mammography')
os.makedirs(model_dir, exist_ok=True)
model = YOLO('yolov8n.pt')
model.save(f'{model_dir}/best.pt')
print(f'✅ Model saved to {model_dir}/best.pt')
EOF

# 5. AI 서비스 실행
cd mammography_ai_service
nohup python app.py > /tmp/mammography-ai.log 2>&1 &

# 6. 확인
sleep 5
curl http://localhost:5004/health
echo "✅ AI 서비스 실행 완료"
```

---

## ⚠️ 주의사항

1. **포트 번호**: AI 서비스는 **5004 포트**를 사용합니다 (5003 아님)
2. **메모리**: YOLO 모델은 최소 2GB RAM 필요
3. **CPU vs GPU**: GPU가 없으면 추론 시간이 오래 걸립니다 (30초~1분)
4. **타임아웃**: Django 설정에서 AI 서비스 타임아웃을 120초로 설정

---

## 🔗 관련 파일

- AI 서비스: `backend/mammography_ai_service/app.py`
- Django API: `backend/mri_viewer/mammography_ai_views.py`
- 모델 경로: `~/models/yolo11_mammography/best.pt`
- 로그: `/tmp/mammography-ai.log`
