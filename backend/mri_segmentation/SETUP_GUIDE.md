# MAMA-MIA Phase 1 Segmentation - 설치 및 사용 가이드

## 📦 패키지 구조

```
MAMA_MIA_DELIVERY_PKG/
├── checkpoints/
│   └── best_model.pth          # 학습된 모델 (필수)
├── src/
│   ├── models/                 # 모델 아키텍처
│   │   ├── __init__.py
│   │   └── swin_unetr_lora.py
│   ├── config.py               # 설정 파일
│   ├── inference_pipeline.py   # 메인 추론 파이프라인
│   ├── inference_preprocess.py # 전처리
│   ├── inference_postprocess.py # 후처리 + DICOM SEG
│   ├── api_server.py           # FastAPI 서버
│   ├── visualize_segmentation.py # 시각화
│   └── requirements.txt        # 의존성 목록
├── sample_data/                # 샘플 데이터
│   └── ISPY2_213913/          # NIfTI 샘플
├── results/                    # 결과 저장 폴더
├── run_demo.py                 # 데모 실행 스크립트
├── README_DELIVERY.md          # 상세 문서
└── RELEASE_NOTES.md            # 릴리즈 노트
```

## 🚀 빠른 시작 (5분 안에 실행)

### 1단계: 압축 해제
```bash
# Windows
압축 파일 우클릭 → "압축 풀기"

# Linux/Mac
unzip MAMA_MIA_DELIVERY_PKG_FINAL.zip
cd MAMA_MIA_DELIVERY_PKG
```

### 2단계: Python 환경 설정
```bash
# Python 3.8 이상 필요
python --version  # 확인

# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3단계: 의존성 설치
```bash
cd src
pip install -r requirements.txt
cd ..
```

**설치 시간**: 약 2-3분 (인터넷 속도에 따라)

### 4단계: 데모 실행
```bash
python run_demo.py
```

**결과 확인**:
- `results/ISPY2_213913_segmentation.nii.gz` - 세그멘테이션 결과
- `results/ISPY2_213913_visualization.png` - 시각화 이미지

---

## 📋 상세 설치 가이드

### A. 시스템 요구사항

#### 최소 사양
- **CPU**: 4 vCPU (Intel/AMD)
- **RAM**: 8GB
- **저장공간**: 5GB
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **Python**: 3.8, 3.9, 3.10, 3.11

#### 권장 사양
- **CPU**: 8+ vCPU
- **RAM**: 16GB
- **GPU**: NVIDIA GPU (CUDA 11.0+) - 선택사항
- **저장공간**: 10GB

### B. 의존성 상세

#### 핵심 라이브러리
```txt
torch>=2.0.0              # PyTorch (딥러닝 프레임워크)
monai[all]>=1.3.0         # 의료 영상 AI 라이브러리
nibabel                   # NIfTI 파일 처리
pydicom>=2.3.0            # DICOM 파일 처리
highdicom>=0.20.0         # DICOM SEG 생성
numpy                     # 수치 연산
scipy                     # 과학 계산
```

#### 선택 라이브러리 (API 서버용)
```txt
fastapi                   # REST API 서버
uvicorn[standard]         # ASGI 서버
python-multipart          # 파일 업로드 처리
```

### C. GPU 사용 시 추가 설정

#### CUDA 설치 (NVIDIA GPU 사용 시)
```bash
# CUDA 11.8 예시 (PyTorch 2.0 호환)
# https://developer.nvidia.com/cuda-downloads

# PyTorch GPU 버전 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### GPU 확인
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

---

## 💻 사용 방법

### 방법 1: 데모 스크립트 (가장 간단)

```bash
python run_demo.py
```

**처리 과정**:
1. 모델 로딩 (CPU/GPU 자동 감지)
2. 샘플 데이터 추론
3. 결과 저장 (`results/` 폴더)
4. 시각화 생성

**예상 시간**:
- CPU: 8-10초
- GPU: 1-2초

### 방법 2: Python 스크립트

```python
import sys
sys.path.insert(0, 'src')

from inference_pipeline import SegmentationInferencePipeline

# 1. 파이프라인 초기화
pipeline = SegmentationInferencePipeline(
    model_path="checkpoints/best_model.pth",
    device="cpu",  # 또는 "cuda"
    threshold=0.5
)

# 2. NIfTI 입력으로 추론
result = pipeline.predict(
    image_path="sample_data/ISPY2_213913",
    output_path="output_segmentation.nii.gz"
)

# 3. 결과 확인
print(f"Tumor Detected: {result['tumor_detected']}")
print(f"Tumor Volume: {result['tumor_volume_voxels']} voxels")
```

### 방법 3: DICOM 입력 + DICOM SEG 출력

```python
# DICOM 폴더 구조 예시:
# dicom_data/
#   ├── seq_0/  (134개 .dcm 파일)
#   ├── seq_1/  (134개 .dcm 파일)
#   ├── seq_2/  (134개 .dcm 파일)
#   └── seq_3/  (134개 .dcm 파일)

pipeline = SegmentationInferencePipeline(
    model_path="checkpoints/best_model.pth",
    device="cpu"
)

result = pipeline.predict(
    image_path="dicom_data",
    output_path="output_seg.dcm",
    output_format="dicom"  # DICOM SEG 생성
)
```

### 방법 4: API 서버 실행

```bash
cd src
python api_server.py
```

**서버 주소**: `http://localhost:8080`

**API 사용 예시**:
```bash
# NIfTI 업로드
curl -X POST "http://localhost:8080/predict" \
  -F "file=@patient_001.nii.gz" \
  -F "output_format=nifti"

# DICOM ZIP 업로드 (Orthanc 연동용)
curl -X POST "http://localhost:8080/predict" \
  -F "file=@dicom_series.zip" \
  -F "output_format=dicom" \
  --output result_seg.dcm
```

---

## 🔧 문제 해결

### Q1: `ModuleNotFoundError: No module named 'monai'`
**해결**: 의존성 재설치
```bash
pip install -r src/requirements.txt
```

### Q2: `CUDA out of memory` (GPU 사용 시)
**해결**: CPU 모드로 전환
```python
pipeline = SegmentationInferencePipeline(
    model_path="checkpoints/best_model.pth",
    device="cpu"  # GPU → CPU
)
```

### Q3: `FileNotFoundError: No .dcm files found`
**해결**: DICOM 폴더 구조 확인
- 폴더 안에 `.dcm` 파일이 있는지 확인
- 하위 폴더 구조 (`seq_0`, `seq_1` 등) 사용 가능

### Q4: 추론 속도가 너무 느림
**해결**:
1. GPU 사용 (`device="cuda"`)
2. 배치 크기 조정 (고급 사용자)
3. CPU 코어 수 확인 (`htop` 또는 작업 관리자)

### Q5: DICOM SEG가 PACS에서 안 보임
**확인사항**:
1. 원본 DICOM에 `FrameOfReferenceUID` 있는지 확인
2. PACS 뷰어가 DICOM SEG 지원하는지 확인
3. 같은 Study/Series에 업로드했는지 확인

---

## 📞 지원

### 로그 확인
```bash
# 상세 로그 출력
python run_demo.py > log.txt 2>&1
```

### 버전 정보
```python
import torch
import monai
print(f"PyTorch: {torch.__version__}")
print(f"MONAI: {monai.__version__}")
```

### 문의
- 개발팀: MAMA-MIA Team
- 버전: 1.0 (2026-01-20)
