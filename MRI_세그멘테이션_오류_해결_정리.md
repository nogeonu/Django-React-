# MRI 세그멘테이션 오류 해결 정리

## 📋 개요
- **기능**: MRI 4-ch DCE-MRI 세그멘테이션 (SwinUNET 모델)
- **모델**: SwinUNETR (4-channel 입력, 3D tumor segmentation)
- **서비스**: Mosec (포트 5006)
- **출력**: DICOM SEG (Multi-frame, 96 frames)
- **기간**: 2026년 1월

---

## 🔴 오류 1: NumberOfFrames는 96인데 첫 프레임 픽셀 값이 모두 0

### 증상
```
DICOM SEG 파일 생성 완료
NumberOfFrames: 96
첫 프레임 PixelData: 모두 0 (검정 화면)
```

### 원인
- 모델 출력 후 후처리 과정에서 문제 발생
- 마스크가 제대로 생성되지 않았거나 0으로 초기화됨
- 디버깅 로그 부족으로 원인 파악 어려움

### 해결 과정
1. **디버깅 로그 추가**: 모델 출력 통계 확인
   ```python
   logger.info(f"모델 출력 - min: {pred_prob.min()}, max: {pred_prob.max()}, mean: {pred_prob.mean()}")
   logger.info(f"종양 픽셀 수: {(pred_mask > 0).sum()}")
   ```

2. **모델 출력 검증**: 
   - 모델이 정상적으로 마스크를 생성하는지 확인
   - 후처리 전후 비교

3. **확인 결과**:
   - 모델 출력은 정상적으로 생성됨
   - DICOM SEG 생성 시 PixelData 인코딩 문제 없음
   - Orthanc UI 표시 문제로 확인 (실제 파일은 정상)

### 해결
- 디버깅 로그로 모델 출력 통계 확인
- DICOM SEG 파일은 정상적으로 생성됨을 확인
- Orthanc UI는 실제 데이터와 무관하게 표시될 수 있음을 확인

### 수정 파일
- `backend/segmentation_mosec.py`: 디버깅 로그 추가

---

## 🔴 오류 2: slice_2d가 None - 4-channel 모드에서 이미지 차원 문제

### 증상
```
slice_2d is None in 4-channel mode, using default (256, 256)
AttributeError: 'NoneType' object has no attribute 'shape'
```

### 원인
- 4-channel 모드에서 `slice_2d`를 추출하는 로직 문제
- 원본 DICOM에서 2D 슬라이스를 추출하지 못함
- 단일 프레임 DICOM이 아닌 경우 처리 로직 오류

### 해결
원본 DICOM의 `Rows`와 `Columns`를 직접 사용:

```python
# 수정 전 (에러)
if slice_2d is None:
    logger.warning(f"slice_2d is None in 4-channel mode, using default (256, 256)")
    h, w = 256, 256  # ❌ 고정값 사용

# 수정 후 (성공) ✅
if slice_2d is None:
    # 원본 DICOM의 차원 사용
    h = original_dicom.Rows
    w = original_dicom.Columns
    logger.info(f"slice_2d is None, using DICOM dimensions: {h}x{w}")
```

### 수정 파일
- `backend/segmentation_mosec.py`: `create_dicom_seg_multiframe` 함수 수정

---

## 🔴 오류 3: Orthanc UI에서 빈 프레임, Instance Number 20 (1이어야 함)

### 증상
```
Orthanc UI:
- # frames: (빈 값)
- Instance Number: 20 (기대: 1)
- SOP Instance UID: (일부만 표시, 잘림)
- 검정 화면 표시
```

### 원인 분석
1. **Instance Number 문제**: 
   - `start_instance_number`가 20으로 설정됨
   - DICOM SEG 생성 시 `InstanceNumber` 필드 설정 오류

2. **빈 프레임 표시**:
   - Orthanc UI 버그 가능성
   - 또는 DICOM SEG 구조 문제

3. **SOP Instance UID 잘림**:
   - Orthanc UI 표시 제한
   - 실제 파일에는 문제 없음

### 해결 과정
1. **디버깅 로그 확인**:
   ```python
   logger.info(f"✅ DICOM SEG 생성: Instance={start_instance_number}, Frames={num_frames}")
   logger.info(f"✅ PixelData 크기: {len(ds.PixelData)} bytes")
   logger.info(f"✅ 종양 픽셀 수: {(mask_array_3d > 0).sum()}")
   ```

2. **Instance Number 수정**:
   ```python
   # 수정 전
   ds.InstanceNumber = str(start_instance_number)  # 20

   # 수정 후 ✅
   ds.InstanceNumber = '1'  # SEG 파일은 항상 1
   ```

3. **검증 결과**:
   - DICOM SEG 파일은 정상적으로 생성됨
   - 96개 프레임 모두 포함
   - PixelData에 정상적인 마스크 데이터 포함
   - Orthanc UI 표시는 버그로 확인

### 해결
- `InstanceNumber`를 항상 '1'로 설정
- Orthanc UI 표시 문제는 실제 데이터와 무관함을 확인
- 디버깅 로그로 실제 데이터 정상 생성 확인

### 수정 파일
- `backend/segmentation_mosec.py`: `create_dicom_seg_multiframe` 함수
- `backend/mri_viewer/segmentation_views.py`: `start_instance_number` 처리

---

## 🔴 오류 4: AttributeError - 'FileDataset' object has no attribute 'SOPClassUID'

### 증상
```
AttributeError: 'FileDataset' object has no attribute 'SOPClassUID'
```

### 원인
- DICOM 파일이 손상되었거나 올바르게 로드되지 않음
- 또는 잘못된 DICOM 파일을 읽으려고 시도
- `pydicom.dcmread()` 실패 후 처리되지 않은 예외

### 해결
- DICOM 파일 검증 로직 추가
- 예외 처리 강화:
  ```python
  try:
      dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
      if not hasattr(dicom, 'SOPClassUID'):
          raise ValueError("Invalid DICOM file: missing SOPClassUID")
  except Exception as e:
      logger.error(f"DICOM 읽기 실패: {str(e)}")
      raise
  ```

### 수정 파일
- `backend/segmentation_mosec.py`: DICOM 파일 검증 로직 추가

---

## 🔴 오류 5: 세그멘테이션 오버레이가 한 시리즈에만 표시됨

### 증상
```
1. 시리즈2를 보고 있음
2. 4개 시리즈 선택 (체크박스)
3. "추론" 버튼 클릭
4. 세그멘테이션 실행 → 시리즈2에만 오버레이 표시
```

### 원인
- 세그멘테이션 결과를 하나의 시리즈에만 저장
- 프론트엔드에서 선택된 4개 시리즈 모두에 매핑하지 않음
- `segmentationFrames` 상태가 단일 시리즈에만 업데이트됨

### 해결
**프론트엔드 수정**: 선택된 4개 시리즈 모두에 세그멘테이션 결과 매핑

```typescript
// 수정 전 (에러)
const handleAiAnalysis = async () => {
  // ... 세그멘테이션 실행
  setSegmentationFrames({ [currentSeriesId]: frames });  // ❌ 한 시리즈만
  setSegmentationStartIndex({ [currentSeriesId]: startIdx });
};

// 수정 후 (성공) ✅
const handleAiAnalysis = async () => {
  // ... 세그멘테이션 실행 (4개 시리즈)
  const newFrames: { [seriesId: string]: any[] } = {};
  const newStartIndex: { [seriesId: string]: number } = {};
  
  selectedSeriesFor4Channel.forEach(seriesIndex => {
    const seriesId = seriesGroups[seriesIndex].series_id;
    newFrames[seriesId] = frames;  // 모든 시리즈에 동일한 프레임 할당
    newStartIndex[seriesId] = startIdx;
  });
  
  setSegmentationFrames(prev => ({ ...prev, ...newFrames }));
  setSegmentationStartIndex(prev => ({ ...prev, ...newStartIndex }));
};
```

### 수정 파일
- `frontend/src/pages/MRIImageDetail.tsx`: `handleAiAnalysis` 함수 수정

---

## 🔴 오류 6: 마스크 위치 불일치 - 좌우 반전 및 슬라이스 인덱스 불일치

### 증상
```
1. 실제 유방은 오른쪽에 있음
2. 마스크는 왼쪽에 표시됨 (좌우 반전)
3. 슬라이스 인덱스 불일치 (앞/뒤 슬라이스에 잘못된 마스크 표시)
```

### 원인
1. **좌우 반전**:
   - DICOM 좌표계와 화면 표시 좌표계 불일치
   - 또는 이미지 반전 변환 누락

2. **슬라이스 인덱스 불일치**:
   - `selectedImageIndex`와 실제 `frameIndex` 매핑 오류
   - `start_slice_index`를 고려하지 않음
   - 슬라이스 번호 계산 오류

### 해결

**1. 좌우 반전 해결**:
```typescript
// 프론트엔드 오버레이 이미지 스타일 수정
<img
  src={`data:image/png;base64,${frameData}`}
  style={{
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    opacity: overlayOpacity,
    transform: 'scaleX(-1)',  // ✅ 좌우 반전
    pointerEvents: 'none',
  }}
/>
```

**2. 슬라이스 인덱스 매핑 수정**:
```typescript
// 수정 전 (에러)
const frameIndex = selectedImageIndex;  // ❌ startIdx 무시

// 수정 후 (성공) ✅
const startIdx = segmentationStartIndex[currentSeriesId] || 0;
const frameIndex = selectedImageIndex - startIdx;

if (frameIndex >= 0 && frameIndex < frames.length) {
  const frameData = frames[frameIndex];
  // 오버레이 표시
}
```

### 수정 파일
- `frontend/src/pages/MRIImageDetail.tsx`: 
  - 오버레이 이미지 `transform: 'scaleX(-1)'` 추가
  - 슬라이스 인덱스 계산 로직 수정

---

## 🔴 오류 7: 마스크가 너무 크거나 유방을 벗어남

### 증상
```
1. 일부 마스크는 정상
2. 일부 마스크는 실제 종양보다 더 크게 표시
3. 일부 마스크는 유방 영역을 벗어남
```

### 근본 원인
**모델 학습 해상도 문제**:
- 모델: **96×96×96**으로 학습됨
- 원본 이미지: **256×256×96** (또는 다른 해상도)
- **다운샘플링 → 추론 → 업샘플링** 과정에서 정보 손실 및 아티팩트

### 해결 과정

**1. 임시 해결 (Post-processing 강화)**:
```python
# 수정 전
pred_mask = (pred_prob > 0.5).astype(np.uint8)  # ❌ 임계값 0.5

# 수정 후 ✅
pred_mask = (pred_prob > 0.7).astype(np.uint8)  # 임계값 증가

# 후처리 강화
def postprocess_mask(mask):
    # 1. 형태학적 침식 (erosion) - 마스크 축소
    mask_eroded = ndimage.binary_erosion(mask, structure=np.ones((3,3,3)))
    
    # 2. 구멍 채우기
    mask_filled = ndimage.binary_fill_holes(mask_eroded)
    
    # 3. 작은 객체 제거 (노이즈 제거)
    labeled, num_features = ndimage.label(mask_filled)
    if num_features > 0:
        sizes = ndimage.sum(mask_filled, labeled, range(1, num_features + 1))
        # 더 강한 필터링: 최소 크기 증가
        min_size = max(100, sizes.max() * 0.1)  # 최소 최대 크기의 10%
        mask_cleaned = np.zeros_like(mask_filled)
        for i in range(1, num_features + 1):
            if sizes[i-1] >= min_size:
                mask_cleaned[labeled == i] = 1
    else:
        mask_cleaned = mask_filled.astype(np.uint8)
    
    return mask_cleaned
```

**2. 업샘플링 제거 검토**:
- 모델 출력을 96×96에서 원본 크기로 업샘플링하지 않음
- 모델을 **256×256**으로 재학습 권장 (근본적 해결)

**3. ROI 크기 수정**:
```python
# 수정 전 (에러)
roi_size = (96, 96, 96)  # ❌ 학습 해상도

# 수정 후 (재학습 후)
roi_size = (256, 256, 256)  # ✅ 원본 해상도 (재학습 필요)
```

### 근본적 해결 방안

**모델 재학습 (256×256)**:
- 현재 모델: 96×96×96 학습
- 권장: 256×256×96 (또는 원본 해상도) 재학습
- 장점:
  - 다운샘플링 정보 손실 없음
  - 업샘플링 아티팩트 없음
  - 더 정확한 마스크 위치
- 단점:
  - 재학습 시간 필요 (3060Ti GPU로 가능)
  - 메모리 사용량 증가

### 임시 해결 상태
- ✅ 임계값 증가 (0.5 → 0.7)
- ✅ 형태학적 침식 추가
- ✅ 작은 객체 필터링 강화
- ⚠️ 근본적 해결은 재학습 필요

### 수정 파일
- `backend/segmentation_mosec.py`: 
  - `postprocess_mask` 함수 강화
  - 임계값 0.5 → 0.7
  - 형태학적 연산 추가

---

## ✅ 최종 해결된 구조

### 데이터 흐름
```
1. 프론트엔드 (React)
   └─ 4개 시리즈 선택 → "추론" 버튼 클릭
   └─ POST /api/mri/segmentation/analyze/
      └─ body: {
           "series_ids": [id1, id2, id3, id4],
           "orthanc_url": "http://localhost:8042",
           "orthanc_auth": ["admin", "admin123"]
         }

2. Django (segmentation_views.py)
   └─ Orthanc에서 4개 시리즈 DICOM 다운로드
   └─ POST http://localhost:5006/inference
      └─ DICOM 데이터 전송 (base64 인코딩)

3. Mosec (segmentation_mosec.py)
   └─ deserialize: JSON → DICOM 바이트 배열
   └─ forward:
      ├─ DICOM → 3D 볼륨 변환 (4개 시퀀스)
      ├─ [4, 96, 96, 96] 입력 준비 (다운샘플링)
      ├─ SwinUNETR 모델 추론
      ├─ 후처리 (임계값 0.7, 침식, 필터링)
      └─ [96, H, W] 마스크 생성
   └─ serialize: DICOM SEG 생성 → Orthanc 업로드
      └─ 반환: seg_instance_id, start_slice_index

4. Django (응답 처리)
   └─ seg_instance_id 반환
   └─ start_slice_index 반환 (슬라이스 매핑용)

5. 프론트엔드
   └─ GET /api/mri/segmentation/instances/{seg_instance_id}/frames/
      └─ 96개 프레임 이미지 (base64 PNG)
   └─ 선택된 4개 시리즈 모두에 오버레이 매핑
   └─ 슬라이스 인덱스 계산: frameIndex = selectedImageIndex - startIdx
   └─ 좌우 반전: transform: 'scaleX(-1)'
```

### 핵심 포인트

1. **4-Channel 입력 처리**
   - 4개 DCE-MRI 시퀀스를 하나의 4-channel 볼륨으로 결합
   - 시간 순서: Pre-contrast → Early → Mid → Late

2. **다운샘플링 문제**
   - 모델은 96×96으로 학습, 원본은 256×256
   - 임시 해결: 후처리 강화
   - 근본 해결: 256×256 재학습 필요

3. **슬라이스 매핑**
   - `start_slice_index`를 사용하여 올바른 프레임 매핑
   - 프론트엔드에서 인덱스 계산 정확히 수행

4. **좌우 반전**
   - DICOM 좌표계와 화면 좌표계 불일치
   - CSS `transform: scaleX(-1)`로 해결

---

## 📊 최종 테스트 결과

### 성공 케이스
- ✅ 4개 시리즈 선택 및 추론 정상 작동
- ✅ 세그멘테이션 오버레이 모든 시리즈에 표시
- ✅ 슬라이스 인덱스 정확히 매핑
- ✅ 좌우 반전 해결
- ✅ DICOM SEG 파일 정상 생성 (96 frames)
- ⚠️ 마스크 크기/위치는 후처리로 부분 개선 (재학습 권장)

### 로그 확인
```bash
# Mosec 로그
sudo journalctl -u mosec-segmentation -f

# Django 로그
sudo journalctl -u gunicorn -f
```

---

## 🔧 설정 파일

### `/etc/systemd/system/mosec-segmentation.service`
```ini
[Unit]
Description=Mosec Segmentation Service
After=network.target

[Service]
Type=simple
User=shrjsdn908
WorkingDirectory=/home/shrjsdn908
ExecStart=/usr/bin/python3 /home/shrjsdn908/segmentation_mosec.py --port 5006 --timeout 300000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 주요 파라미터
- `--port 5006`: Mosec 서비스 포트
- `--timeout 300000`: 타임아웃 300초 (MRI 추론 시간 고려)
- `max_batch_size`: 1 (MRI는 배치 처리 불가)

---

## 📝 교훈

1. **해상도 불일치 문제**
   - 학습 해상도와 추론 해상도가 다르면 정확도 저하
   - 가능하면 원본 해상도로 학습 권장
   - 불가피한 경우 후처리로 보완

2. **좌표계 이해**
   - DICOM 좌표계와 화면 좌표계 차이
   - 필요시 변환 적용

3. **슬라이스 인덱스 매핑**
   - `start_index`를 고려한 정확한 매핑 필요
   - 프론트엔드-백엔드 인덱스 일치 확인

4. **디버깅 로그의 중요성**
   - 각 단계별 상세 로그로 문제 빠르게 파악
   - 모델 출력 통계, 후처리 결과 확인

5. **점진적 해결**
   - 임시 해결 (후처리 강화) → 근본 해결 (재학습)
   - 단계적으로 문제 해결

6. **UI 버그 vs 실제 버그**
   - Orthanc UI 표시 문제와 실제 데이터 문제 구분
   - 실제 파일 검증으로 확인

---

## 🎯 최종 상태

✅ **대부분의 오류 해결 완료**
✅ **4개 시리즈 세그멘테이션 정상 작동**
✅ **오버레이 표시 정상 (좌우 반전, 슬라이스 매핑 해결)**
✅ **DICOM SEG 파일 정상 생성**
⚠️ **마스크 크기/위치는 부분 개선** (256×256 재학습 권장)

---

## 🔮 향후 개선 사항

1. **모델 재학습 (256×256)**
   - GPU: 3060Ti (가능)
   - 정확도 향상 기대

2. **실시간 슬라이스 스크롤 최적화**
   - 프레임 프리로딩
   - 캐싱 전략

3. **다중 시리즈 동시 처리**
   - 병렬 처리로 추론 시간 단축

4. **오버레이 투명도 조절**
   - 사용자가 직접 조절 가능

---

**작성일**: 2026년 1월 10일
**작성자**: AI Assistant
**검증자**: 사용자 (실제 테스트 완료)

