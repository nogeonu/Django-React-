# Phase 1 Segmentation - GCP Deployment

## GCP 배포 가이드

### 필수 요구사항
- Google Cloud SDK 설치
- Docker 설치
- GCP 프로젝트 생성

---

## 배포 방법

### 0. 환경 설정 (중요!)
```bash
# ⚠️ pydicom 버전 주의!
# pydicom-seg는 pydicom 2.x만 지원
# pydicom 3.x는 호환 안 됨

pip install pydicom==2.4.4
pip install pydicom-seg==0.4.1
pip install highdicom SimpleITK

# 또는 requirements.txt 사용
pip install -r requirements.txt
```

### 1. GCP 프로젝트 설정
```bash
# GCP 로그인
gcloud auth login

# 프로젝트 설정
gcloud config set project YOUR_PROJECT_ID

# Container Registry 활성화
gcloud services enable containerregistry.googleapis.com
gcloud services enable run.googleapis.com
```

### 2. Docker 이미지 빌드 및 푸시
```bash
# 이미지 빌드
docker build -t gcr.io/YOUR_PROJECT_ID/phase1-segmentation:latest .

# GCP Container Registry에 푸시
docker push gcr.io/YOUR_PROJECT_ID/phase1-segmentation:latest
```

### 3. Cloud Run 배포
```bash
gcloud run deploy phase1-segmentation \
    --image gcr.io/YOUR_PROJECT_ID/phase1-segmentation:latest \
    --platform managed \
    --region us-central1 \
    --memory 8Gi \
    --cpu 4 \
    --timeout 300 \
    --allow-unauthenticated
```

---

## API 사용법

### Health Check
```bash
curl https://YOUR_SERVICE_URL/health
```

### Segmentation 예측
```bash
curl -X POST https://YOUR_SERVICE_URL/predict \
    -F "file=@patient_mri.nii.gz" \
    -F "output_format=nifti" \
    --output segmentation.nii.gz
```

---

## 로컬 테스트

### Docker로 로컬 실행
```bash
# 이미지 빌드
docker build -t phase1-segmentation .

# 컨테이너 실행
docker run -p 8080:8080 phase1-segmentation

# 테스트
curl http://localhost:8080/health
```

### Python으로 직접 실행
```bash
python api_server.py
```

---

## 성능 최적화

### GPU 사용 (선택)
Cloud Run은 GPU를 지원하지 않으므로, GPU가 필요하면 GKE 사용:
```bash
# GKE 클러스터 생성 (GPU 노드)
gcloud container clusters create phase1-cluster \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --zone us-central1-a
```

### 메모리 설정
- 최소: 4GB (추론만)
- 권장: 8GB (안정적 운영)

---

## 비용 예상

**Cloud Run** (권장):
- 메모리 8GB, CPU 4
- 요청당 약 $0.01-0.05
- 월 1000건: ~$10-50

**GKE** (GPU 필요 시):
- T4 GPU 인스턴스
- 월 ~$300-500

---

## 모니터링

### 로그 확인
```bash
gcloud run logs read phase1-segmentation --limit 50
```

### 메트릭 확인
GCP Console → Cloud Run → phase1-segmentation → Metrics

---

## 문제 해결

### 메모리 부족
→ `--memory 16Gi`로 증가

### 타임아웃
→ `--timeout 600`으로 증가

### Cold Start 느림
→ `--min-instances 1` 설정 (비용 증가)

---

## 보안

### 인증 추가
```bash
# 인증 필요하도록 재배포
gcloud run deploy phase1-segmentation \
    --no-allow-unauthenticated
```

### API Key 사용
`api_server.py`에 API Key 검증 로직 추가
