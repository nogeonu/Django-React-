# Phase 1: Breast Tumor Segmentation

**완전한 Phase 1 Segmentation 배포 패키지**

## 📦 포함 내용

### 학습 관련
- `train_segmentation.py` - 학습 스크립트
- `dataset.py` - 데이터셋 클래스
- `config.py` - 설정 파일
- `models/` - SwinUNETR + LoRA 모델 정의

### 추론 관련
- `inference_pipeline.py` - 통합 추론 파이프라인
- `inference_preprocess.py` - 전처리 모듈
- `inference_postprocess.py` - 후처리 모듈 (NIfTI/DICOM 출력)
- `visualize_segmentation.py` - 시각화 모듈

### 문서
- `INFERENCE_README.md` - 추론 가이드
- `requirements.txt` - 필수 라이브러리

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 추론 실행
```bash
# NIfTI 출력
python inference_pipeline.py \
    --model best_model.pth \
    --input patient_folder/ \
    --output result.nii.gz

# DICOM 출력 (PACS 통합용)
python inference_pipeline.py \
    --model best_model.pth \
    --input patient_folder/ \
    --output result.dcm \
    --format dicom \
    --dicom-dir patient_folder/

# 시각화
python visualize_segmentation.py \
    --image patient_folder/ \
    --segmentation result.nii.gz \
    --output visualization.png
```

### 3. 학습 (선택)
```bash
python train_segmentation.py
```

---

## 📊 모델 성능

- **Validation Dice**: 0.7625
- **EMA Dice**: 0.7655
- **학습 Epochs**: 116 (Early Stopping)
- **입력**: 4채널 DCE-MRI (128³)
- **출력**: Binary Segmentation Mask

---

## 🏥 배포 시나리오

### 병원 PACS 통합
```python
from inference_pipeline import SegmentationInferencePipeline

pipeline = SegmentationInferencePipeline("best_model.pth")
result = pipeline.predict(
    "patient_dicom_folder/",
    output_format='dicom',
    output_path="tumor.dcm"
)
```

### 연구용 분석
```python
result = pipeline.predict(
    "patient_nifti.nii.gz",
    output_format='nifti',
    output_path="segmentation.nii.gz"
)
```

---

## 📝 주요 특징

- ✅ **4채널 DCE-MRI 입력**
- ✅ **1.5mm spacing 표준화**
- ✅ **128³ 패치 크기**
- ✅ **NIfTI/DICOM 출력 지원**
- ✅ **원본 spacing 복원 (선택)**
- ✅ **시각화 PNG 생성**

---

## 📂 파일 구조

```
Phase1_Segmentation/
├── train_segmentation.py      # 학습
├── dataset.py                  # 데이터셋
├── config.py                   # 설정
├── inference_pipeline.py       # 추론 파이프라인
├── inference_preprocess.py     # 전처리
├── inference_postprocess.py    # 후처리
├── visualize_segmentation.py   # 시각화
├── models/                     # 모델 정의
│   ├── segmentation.py
│   └── lora.py
├── requirements.txt
├── INFERENCE_README.md
└── README.md (이 파일)
```

---

## 🔧 설정 변경

`config.py`에서 다음 설정 가능:
- `DATA_ROOT`: 데이터 경로
- `PATCH_SIZE`: 패치 크기 (기본: 128³)
- `SPACING`: 리샘플링 spacing (기본: 1.5mm)
- `BATCH_SIZE`: 배치 크기
- `NUM_WORKERS`: 데이터 로더 워커 수

---

## 📞 문의

Phase 1 Segmentation 전용 패키지입니다.
Phase 2-4는 별도 패키지를 참조하세요.
