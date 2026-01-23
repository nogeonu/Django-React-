# 연구실 컴퓨터 MRI 추론 패키지

## 📦 패키지 내용

이 패키지는 연구실 컴퓨터에서 MRI 세그멘테이션 추론을 실행하기 위한 모든 파일을 포함합니다.

### 포함된 파일

```
연구실_컴퓨터_추론_패키지/
├── README.md                           # 이 파일
├── 연구실_컴퓨터_추론_구조_변경안.md       # 전체 구조 설명
├── mri_segmentation/
│   ├── local_inference.py              # ⭐ 수동 추론 스크립트
│   ├── local_inference_worker.py       # ⭐ 자동 워커 스크립트
│   ├── env.example                     # 환경 변수 예시
│   ├── README_LOCAL_INFERENCE.md       # 완전 가이드
│   ├── 연구실_컴퓨터_실행_가이드.md        # 빠른 시작 가이드
│   ├── 변경사항_요약.md                 # 변경 사항 요약
│   ├── systemd/
│   │   └── mri-inference-worker.service # systemd 서비스 파일
│   └── src/
│       ├── inference_pipeline.py       # 추론 파이프라인
│       ├── inference_preprocess.py     # 전처리
│       ├── inference_postprocess.py    # 후처리
│       ├── config.py                   # 설정
│       ├── requirements.txt            # Python 의존성
│       └── models/                     # 모델 아키텍처
```

---

## 🚀 빠른 시작 (3단계만!)

### 1️⃣ 압축 해제
```bash
cd ~
unzip 연구실_컴퓨터_추론_패키지.zip
cd 연구실_컴퓨터_추론_패키지/mri_segmentation
```

### 2️⃣ 환경 설정
```bash
# Python 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r src/requirements.txt

# GPU 버전 (선택사항, 권장)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 환경 변수 설정
cp env.example .env
nano .env  # ORTHANC_PASSWORD 수정
```

### 3️⃣ 자동 워커 실행 (권장) ⭐

**프론트엔드에서 "AI 분석" 버튼을 누르면 자동으로 추론됩니다!**

```bash
# ⚠️ 먼저 모델 파일을 src/ 디렉토리에 복사해야 합니다!
# scp user@server:/path/to/best_model.pth src/

# 자동 워커 실행 (백그라운드)
nohup python local_inference_worker.py > worker.log 2>&1 &

# 또는 systemd 서비스 (프로덕션 권장)
sudo cp systemd/mri-inference-worker.service /etc/systemd/system/
# 파일 내 경로 수정 후:
sudo systemctl daemon-reload
sudo systemctl enable mri-inference-worker
sudo systemctl start mri-inference-worker
```

**✅ 워커가 실행 중이면 프론트엔드에서 자동으로 추론됩니다!**

### 4️⃣ 수동 실행 (테스트용)

```bash
# 수동으로 추론 실행 (테스트용)
python local_inference.py \
    --series-ids series1 series2 series3 series4
```

---

## ⚠️ 중요: 모델 파일 필요

**추론 실행 전에 반드시 모델 파일을 복사해야 합니다!**

```bash
# GCP 서버에서 모델 파일 복사
scp user@34.42.223.43:/srv/django-react/app/backend/mri_segmentation/src/best_model.pth src/

# 또는 다른 위치에서 복사
cp /path/to/best_model.pth src/

# 모델 파일 확인
ls -lh src/best_model.pth
# 약 500MB-1GB 크기여야 합니다
```

---

## 📚 상세 가이드

### 환경 변수 설정 (.env 파일)

```bash
ORTHANC_URL=http://34.42.223.43:8042
ORTHANC_USER=admin
ORTHANC_PASSWORD=admin123  # ⚠️ 실제 비밀번호로 변경!
MODEL_PATH=src/best_model.pth
DEVICE=cuda  # 또는 cpu
THRESHOLD=0.5
```

### 수동 추론 실행

```bash
# 기본 실행
python local_inference.py \
    --series-ids "id1" "id2" "id3" "id4"

# GPU 사용 (권장)
python local_inference.py \
    --device cuda \
    --series-ids "id1" "id2" "id3" "id4"

# 임계값 조정
python local_inference.py \
    --threshold 0.7 \
    --series-ids "id1" "id2" "id3" "id4"
```

### 자동 워커 실행 (프론트엔드 자동 연동) ⭐

**워커가 실행 중이면 프론트엔드에서 "AI 분석" 버튼을 누르면 자동으로 추론됩니다!**

```bash
# 포그라운드 (테스트용)
python local_inference_worker.py

# 백그라운드 (권장)
nohup python local_inference_worker.py > worker.log 2>&1 &

# systemd 서비스 (프로덕션 권장)
sudo cp systemd/mri-inference-worker.service /etc/systemd/system/
# 파일 내 경로 수정 후:
sudo systemctl daemon-reload
sudo systemctl enable mri-inference-worker
sudo systemctl start mri-inference-worker

# 워커 상태 확인
ps aux | grep local_inference_worker
tail -f worker.log
```

**워커가 실행 중인지 확인:**
```bash
# 프로세스 확인
ps aux | grep local_inference_worker

# 로그 확인
tail -f worker.log

# systemd 서비스 확인
sudo systemctl status mri-inference-worker
```

**자동화 설정 상세 가이드:** `자동화_설정_가이드.md` 참고

---

## 🔧 문제 해결

### Q1: 모듈을 찾을 수 없음
```bash
pip install -r src/requirements.txt
```

### Q2: 모델 파일 없음
```bash
# 모델 파일을 src/ 디렉토리에 복사
scp user@server:/path/to/best_model.pth src/
```

### Q3: Orthanc 연결 실패
```bash
# 네트워크 확인
ping 34.42.223.43

# Orthanc 상태 확인
curl -u admin:password http://34.42.223.43:8042/system
```

### Q4: GPU 인식 안 됨
```bash
# CUDA 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📊 시스템 요구사항

### 최소
- Python 3.8+
- 8GB RAM
- 20GB 디스크

### 권장
- Python 3.10
- NVIDIA GPU (RTX 3090+)
- 16GB RAM
- 50GB 디스크
- 안정적인 네트워크 (GCP Orthanc 접근)

---

## 📖 문서

- `자동화_설정_가이드.md` - ⭐ **자동화 설정 가이드 (프론트엔드 연동)**
- `README_LOCAL_INFERENCE.md` - 완전한 설치/설정/사용 가이드
- `연구실_컴퓨터_실행_가이드.md` - 빠른 시작 가이드
- `변경사항_요약.md` - 변경 사항 요약
- `연구실_컴퓨터_추론_구조_변경안.md` - 전체 구조 및 배경

## 🎯 자동화 기능 (HTTP API 방식)

### 프론트엔드 자동 연동

**프론트엔드에서 "AI 분석" 버튼을 누르면 자동으로 연구실 컴퓨터에서 추론됩니다!**

#### ✅ HTTP API 방식의 장점
- ✅ **공유 디렉토리 불필요**
- ✅ **연구실 내부 IP 불필요**
- ✅ **인터넷 연결만 있으면 됩니다!**

#### 동작 방식
1. 프론트엔드에서 "AI 분석" 버튼 클릭
2. Django가 자동으로 연구실 컴퓨터 요청 생성
3. 워커가 30초마다 **HTTP API로 요청 확인** (폴링)
4. 워커가 자동 처리 (DICOM 다운로드 → 추론 → 업로드)
5. 워커가 **HTTP API로 결과 업로드**
6. 프론트엔드에 결과 표시

#### 설정 방법

**연구실 컴퓨터:**
```bash
# .env 파일 설정
DJANGO_API_URL=http://34.42.223.43/api/mri

# 워커 실행
python local_inference_worker.py
```

**GCP Django 서버:**
```bash
# 환경 변수 설정
export USE_LOCAL_INFERENCE=true
sudo systemctl restart gunicorn
```

**상세 가이드:** 
- `자동화_설정_가이드.md` - 기본 가이드
- `HTTP_API_방식_설정_가이드.md` - ⭐ HTTP API 방식 상세 가이드

---

## ✅ 설치 체크리스트

설치 전:
- [ ] Python 3.8+ 설치 확인
- [ ] pip 설치 확인
- [ ] 네트워크 연결 (GCP Orthanc 접근 가능)
- [ ] GPU 드라이버 설치 (GPU 사용 시)

설치:
- [ ] 압축 해제 완료
- [ ] 가상환경 생성 완료
- [ ] 의존성 설치 완료
- [ ] **모델 파일 복사 완료** ⚠️ 중요!
- [ ] `.env` 파일 설정 완료

테스트:
- [ ] Orthanc 연결 테스트 성공
- [ ] 수동 추론 실행 성공
- [ ] 결과가 Orthanc에 업로드됨 확인

---

**버전**: 1.0.0
**생성일**: 2026년 1월
