# MAMA-MIA Phase 1 Segmentation - Final Delivery Package

## 📦 패키지 정보
- **버전**: v1.0 (DICOM SEG 완전 지원)
- **생성일**: 2026-01-20
- **테스트 완료**: ✅ NIfTI 및 DICOM 입력 모두 검증

## ✅ 검증 완료 기능

### 1. 입력 형식 지원
- ✅ **NIfTI 파일** (`.nii.gz`) - 4개 시퀀스 파일
- ✅ **DICOM 시리즈** - 폴더 구조 자동 감지
  - 단일 폴더 (모든 `.dcm` 파일)
  - 하위 폴더 구조 (`seq_0`, `seq_1`, `seq_2`, `seq_3`)

### 2. 추론 성능
- **CPU 모드**: 8~11초 (4 vCPU 기준)
- **GPU 모드**: 1~2초 (RTX 3060 기준)
- **메모리**: 약 2GB RAM 사용

### 3. DICOM SEG 생성
- ✅ **Highdicom 라이브러리** 사용
- ✅ **Spatial Sorting** (Z축 물리적 위치 기준 정렬)
- ✅ **Sparse Encoding** (종양 있는 슬라이스만 저장)
- ✅ **원본 해상도 복원** (256x256x134)
- ✅ **Frame of Reference UID** 정합

### 4. 전처리/후처리
- ✅ 1.5mm 등방성 리샘플링
- ✅ RAS 방향 정렬
- ✅ MONAI `Invertd`를 통한 원본 공간 복원
- ✅ Morphological 후처리 (LCC, Hole Filling)

## 📊 테스트 결과 (ISPY2_213913)

### NIfTI 입력 테스트
- 입력: 4개 NIfTI 파일 (256x256x134)
- 추론 시간: 8.51초
- 종양 탐지: 11,122 voxels
- 출력: NIfTI 세그멘테이션 + PNG 시각화

### DICOM 입력 테스트
- 입력: 4개 DICOM 시리즈 (536 파일)
- 추론 시간: 11.49초
- 종양 탐지: 9,705 voxels
- 출력: **DICOM SEG (19 frames, 256x256)**

## 🚀 사용 방법

### 기본 사용 (NIfTI)
```bash
cd MAMA_MIA_DELIVERY_PKG
python run_demo.py
```

### DICOM SEG 생성
```python
from src.inference_pipeline import SegmentationInferencePipeline

pipeline = SegmentationInferencePipeline(
    model_path="checkpoints/best_model.pth",
    device="cpu"
)

result = pipeline.predict(
    "path/to/dicom/folder",
    output_path="output.dcm",
    output_format="dicom"
)
```

## 📁 주요 파일

### 핵심 코드
- `src/inference_pipeline.py` - 전체 파이프라인
- `src/inference_preprocess.py` - 전처리 (DICOM/NIfTI 자동 감지)
- `src/inference_postprocess.py` - 후처리 및 DICOM SEG 생성
- `src/api_server.py` - FastAPI 서버 (ZIP 업로드 지원)

### 모델
- `checkpoints/best_model.pth` - SwinUNETR + LoRA (15.7M params)

### 문서
- `README_DELIVERY.md` - 상세 사용 가이드
- `DEPLOYMENT.md` - GCP 배포 가이드

## 🔧 의존성
- Python 3.8+
- PyTorch 2.0+
- MONAI 1.3+
- **highdicom 0.20+** (DICOM SEG 생성)
- pydicom 2.3+

## ⚠️ 중요 사항

### DICOM SEG 생성 시 필수 조건
1. 입력이 **DICOM 폴더**여야 함 (NIfTI는 불가)
2. 원본 DICOM 파일에 다음 태그 필요:
   - `ImagePositionPatient`
   - `ImageOrientationPatient`
   - `PixelSpacing`
   - `FrameOfReferenceUID`

### Orthanc 연동 워크플로우
1. Orthanc에서 DICOM 시리즈 다운로드 (ZIP)
2. ZIP 업로드 → API 서버 (`/predict?output_format=dicom`)
3. DICOM SEG 파일 수신
4. Orthanc에 업로드 (`POST /instances`)
5. PACS 뷰어에서 오버레이 확인

## 📞 문의
- 개발자: MAMA-MIA Team
- 버전: 1.0 (2026-01-20)
