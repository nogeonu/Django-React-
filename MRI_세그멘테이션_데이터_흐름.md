# MRI 세그멘테이션 데이터 흐름 상세 정리

## 📋 개요
- **기능**: MRI 4-ch DCE-MRI 세그멘테이션 (Tumor Segmentation)
- **모델**: SwinUNETR (4-channel 입력, 3D segmentation)
- **서비스 구조**: Django → Mosec (instance_ids만 전송) → Mosec이 Orthanc에서 직접 DICOM 다운로드 → 세그멘테이션 → Orthanc 업로드 → Django → 프론트엔드
- **포트**: Mosec 5006, Orthanc 8042
- **아키텍처**: 신버전 (instance_ids만 전송, 413 에러 방지)

---

## 🔄 전체 데이터 흐름도

```
[프론트엔드 React]
    │
    │ 1. 4개 시리즈 선택
    │ 2. "AI 추론" 버튼 클릭
    │
    ▼
[Django - segmentation_views.py]
    │ segment_series()
    │
    │ 3. Orthanc에서 4개 시리즈의 Instance ID 수집
    │    ├─ GET /orthanc/series/{series_id}
    │    └─ 각 시리즈의 Instances 목록 가져오기
    │
    │ 4. 중앙 96개 슬라이스 선택
    │    └─ start_idx = (total_slices - 96) // 2
    │
    │ 5. Mosec에 요청 전송 (instance_ids만 전송)
    │    POST http://localhost:5006/inference
    │    Body: {
    │      "orthanc_instance_ids": [
    │        [instance_id_1_1, instance_id_1_2, ..., instance_id_1_96],  // 시퀀스 1
    │        [instance_id_2_1, instance_id_2_2, ..., instance_id_2_96],  // 시퀀스 2
    │        [instance_id_3_1, instance_id_3_2, ..., instance_id_3_96],  // 시퀀스 3
    │        [instance_id_4_1, instance_id_4_2, ..., instance_id_4_96]   // 시퀀스 4
    │      ],
    │      "orthanc_url": "http://localhost:8042",
    │      "orthanc_auth": ["admin", "admin123"],
    │      "seg_series_uid": "...",
    │      "original_series_id": "...",
    │      "start_instance_number": 20
    │    }
    │    ※ DICOM 파일은 전송하지 않음 (instance_ids만, ~몇 KB)
    │
    ▼
[Mosec - segmentation_mosec.py]
    │ SegmentationWorker
    │
    │ 6. deserialize(): JSON → Python dict
    │    └─ orthanc_instance_ids 추출
    │
    │ 7. Orthanc API 직접 호출 (Mosec 내부에서)
    │    ├─ 각 instance_id에 대해:
    │    │  └─ GET /orthanc/instances/{instance_id}/file
    │    │     └─ DICOM 바이트 다운로드 (~19MB × 96슬라이스 × 4시퀀스)
    │    └─ base64 인코딩 (내부 처리)
    │
    │ 8. forward():
    │    ├─ 다운로드한 DICOM 파일들을 3D 볼륨으로 변환
    │    │  ├─ base64 디코딩 → DICOM 바이트
    │    │  └─ pydicom.dcmread() → pixel_array
    │    ├─ 각 시퀀스를 [D, H, W] 형태로 변환 (D=96, H=256, W=256)
    │    ├─ 4개 시퀀스 결합 → [4, 96, 96, 96] (다운샘플링)
    │    ├─ SwinUNETR 모델 추론
    │    │  └─ sliding_window_inference (roi_size=(96,96,96))
    │    ├─ 후처리
    │    │  ├─ 임계값 0.7 적용
    │    │  ├─ 형태학적 침식 (erosion)
    │    │  └─ 작은 객체 필터링
    │    └─ [96, H, W] 마스크 생성
    │
    │ 9. DICOM SEG 생성
    │    └─ create_dicom_seg_multiframe()
    │       ├─ Multi-frame DICOM SEG 구조 생성
    │       ├─ 96개 프레임을 PixelData로 결합
    │       └─ InstanceNumber = '1', NumberOfFrames = 96
    │
    │ 10. Orthanc에 업로드
    │     └─ POST /orthanc/instances
    │        └─ 반환: seg_instance_id
    │
    │ 11. serialize(): dict → JSON bytes
    │     └─ {"seg_instance_id": "...", "start_slice_index": 0}
    │
    ▼
[Django - segmentation_views.py]
    │
    │ 12. Mosec 응답 처리
    │     └─ seg_instance_id, start_slice_index 추출
    │
    │ 13. 응답 반환
    │     └─ {
    │          "success": true,
    │          "seg_instance_id": "...",
    │          "start_slice_index": 19,  // 실제 시작 인덱스
    │          "end_slice_index": 114,
    │          "num_frames": 96
    │        }
    │
    ▼
[프론트엔드 React]
    │ MRIImageDetail.tsx
    │
    │ 14. 세그멘테이션 결과 저장
    │     └─ setSeriesSegmentationResults({
    │          [seriesId1]: {seg_instance_id, start_slice_index},
    │          [seriesId2]: {seg_instance_id, start_slice_index},
    │          [seriesId3]: {seg_instance_id, start_slice_index},
    │          [seriesId4]: {seg_instance_id, start_slice_index}
    │        })
    │
    │ 15. 슬라이스 변경 시 프레임 로드
    │     └─ useEffect(() => {
    │          if (showSegmentationOverlay) {
    │            loadSegmentationFrames(currentSeriesId);
    │          }
    │        }, [selectedImageIndex])
    │
    │ 16. 프레임 요청
    │     └─ GET /api/mri/segmentation/instances/{seg_instance_id}/frames/
    │        ?frame_index={frameIndex}
    │
    ▼
[Django - segmentation_views.py]
    │ get_segmentation_frames()
    │
    │ 17. Orthanc에서 SEG 프레임 추출
    │     ├─ GET /orthanc/instances/{seg_instance_id}/file
    │     ├─ DICOM SEG 파일 다운로드
    │     ├─ PixelData에서 frame_index 번째 프레임 추출
    │     └─ PNG로 변환 (base64)
    │
    │ 18. 응답 반환
    │     └─ {"frame_data": "data:image/png;base64,..."}
    │
    ▼
[프론트엔드 React]
    │
    │ 19. 오버레이 표시
    │     └─ <img
    │          src={frameData}
    │          style={{
    │            transform: 'scaleX(-1)',  // 좌우 반전
    │            opacity: overlayOpacity
    │          }}
    │        />
    │
    │ 20. 원본 MRI 이미지 위에 오버레이 표시
```

---

## 📝 단계별 상세 설명

### 1단계: 프론트엔드 - 사용자 액션

**파일**: `frontend/src/pages/MRIImageDetail.tsx`

```typescript
// 4개 시리즈 선택
const [selectedSeriesFor4Channel, setSelectedSeriesFor4Channel] = useState<number[]>([]);

// "AI 추론" 버튼 클릭
const handleAiAnalysis = async () => {
  if (selectedSeriesFor4Channel.length !== 4) {
    toast({ title: "4개 시리즈를 선택해주세요" });
    return;
  }
  
  setAiAnalyzing(true);
  try {
    // 2단계로 이동
    const response = await fetch('/api/mri/segmentation/analyze/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        series_ids: selectedSeriesFor4Channel.map(idx => 
          seriesGroups[idx].series_id
        )
      })
    });
    
    const data = await response.json();
    // 12단계로 이동
  } catch (error) {
    // 에러 처리
  }
};
```

**데이터 형식**:
```json
{
  "series_ids": [
    "2d3aba01-388e3a29-38c2bec0-3bbae0ef-17be8283",
    "series2-uid",
    "series3-uid",
    "series4-uid"
  ]
}
```

---

### 3-5단계: Django - Orthanc에서 Instance ID 수집 및 Mosec 요청

**파일**: `backend/mri_viewer/segmentation_views.py`

```python
@api_view(['POST'])
def segment_series(request, series_id):
    """
    시리즈 전체를 3D 세그멘테이션하고 Orthanc에 저장 (4-channel, 96 슬라이스)
    """
    try:
        # 1. 요청 데이터 파싱
        sequence_series_ids = request.data.get("sequence_series_ids", [])
        
        if len(sequence_series_ids) != 4:
            return Response({
                "success": False,
                "error": "4개 시리즈가 모두 필요합니다."
            }, status=400)
        
        # 2. Orthanc 클라이언트 초기화
        client = OrthancClient()
        
        # 3. 메인 시리즈 정보 확인 (슬라이스 수)
        main_series_info = client.get(f"/series/{series_id}")
        main_instances = main_series_info.get("Instances", [])
        total_slices = len(main_instances)
        
        if total_slices < 96:
            return Response({
                "success": False,
                "error": f"슬라이스 수가 부족합니다 (최소 96개 필요, 현재 {total_slices}개)"
            }, status=400)
        
        # 4. 중앙 96개 슬라이스 선택
        start_idx = (total_slices - 96) // 2
        end_idx = start_idx + 96
        
        logger.info(f"📍 슬라이스 선택: {start_idx}~{end_idx-1}번 (중앙 96개)")
        
        # 5. 4개 시퀀스에서 각각 96개 슬라이스의 Instance ID 수집
        orthanc_instance_ids = []  # [4][96] 형태
        
        for seq_idx, current_seq_series_id in enumerate(sequence_series_ids):
            seq_info = client.get(f"/series/{current_seq_series_id}")
            seq_instances = seq_info.get("Instances", [])
            
            if len(seq_instances) < 96:
                return Response({
                    "success": False,
                    "error": f"시퀀스 {current_seq_series_id}의 슬라이스가 부족합니다"
                }, status=400)
            
            # 같은 범위에서 96개 선택
            selected_instances = seq_instances[start_idx:end_idx]
            orthanc_instance_ids.append(selected_instances)  # Instance ID 목록만 저장
            
            logger.info(f"✅ 시퀀스 {seq_idx+1}/4 Instance ID 수집 완료: {len(selected_instances)}개")
        
        # 6. Mosec에 Instance ID 목록만 전송 (작은 payload, 몇 KB)
        seg_series_uid = generate_uid()
        
        payload = {
            "orthanc_instance_ids": orthanc_instance_ids,  # [4][96] Instance ID 목록
            "orthanc_url": ORTHANC_URL,
            "orthanc_auth": [ORTHANC_USER, ORTHANC_PASSWORD],
            "seg_series_uid": seg_series_uid,
            "original_series_id": series_id,
            "start_instance_number": start_idx + 1
        }
        
        logger.info(f"📦 Payload 크기: {len(orthanc_instance_ids)}개 시퀀스")
        
        # 7. Mosec에 요청 전송
        seg_response = requests.post(
            f"{SEGMENTATION_API_URL}/inference",
            json=payload,
            timeout=2400  # 40분 (Orthanc 다운로드 + 세그멘테이션 + 업로드)
        )
        
        seg_response.raise_for_status()
        result = seg_response.json()
        
        # 12단계: Mosec 응답 처리
        return Response({
            'success': True,
            'series_id': series_id,
            'total_slices': 96,
            'start_slice_index': start_idx,
            'end_slice_index': end_idx - 1,
            'seg_instance_id': result.get('seg_instance_id'),
            'tumor_ratio_percent': result.get('tumor_ratio_percent', 0),
            'saved_to_orthanc': result.get('saved_to_orthanc', False)
        })
        
    except Exception as e:
        logger.error(f"세그멘테이션 분석 실패: {str(e)}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)
```

**Mosec 요청 데이터 형식**:
```json
{
  "orthanc_instance_ids": [
    [
      "instance_id_1_1", "instance_id_1_2", ..., "instance_id_1_96"
    ],
    [
      "instance_id_2_1", "instance_id_2_2", ..., "instance_id_2_96"
    ],
    [
      "instance_id_3_1", "instance_id_3_2", ..., "instance_id_3_96"
    ],
    [
      "instance_id_4_1", "instance_id_4_2", ..., "instance_id_4_96"
    ]
  ],
  "orthanc_url": "http://localhost:8042",
  "orthanc_auth": ["admin", "admin123"],
  "seg_series_uid": "1.2.826.0.1.3680043.8.498...",
  "original_series_id": "series_id",
  "start_instance_number": 20
}
```

**데이터 크기**: ~몇 KB (Instance ID 목록만)

**핵심 포인트**:
- ✅ DICOM 파일을 Django에서 Mosec으로 전송하지 않음
- ✅ instance_ids만 전송 (413 Request Entity Too Large 에러 방지)
- ✅ Mosec이 Orthanc에서 직접 다운로드

---

### 6-11단계: Mosec - Orthanc에서 DICOM 다운로드 및 세그멘테이션 추론

**파일**: `backend/segmentation_mosec.py`

```python
class SegmentationWorker(Worker):
    def deserialize(self, data: bytes) -> dict:
        """요청 데이터 역직렬화 (Orthanc API 방식)"""
        json_data = json.loads(data.decode('utf-8'))
        
        logger.info(f"📥 수신한 데이터 키: {list(json_data.keys())}")
        
        # Orthanc Instance ID 목록이 있으면 Orthanc API로 다운로드
        if "orthanc_instance_ids" in json_data:
            orthanc_url = json_data["orthanc_url"]
            orthanc_auth = tuple(json_data["orthanc_auth"])
            
            logger.info(f"📥 Orthanc에서 데이터 다운로드 중: {orthanc_url}")
            logger.info(f"📊 총 {len(json_data['orthanc_instance_ids'])}개 시퀀스, 각 {len(json_data['orthanc_instance_ids'][0])}개 슬라이스")
            
            sequences_3d = []
            for seq_idx, seq_instances in enumerate(json_data["orthanc_instance_ids"]):
                slices_data = []
                for slice_idx, instance_id in enumerate(seq_instances):
                    # Orthanc API로 DICOM 파일 다운로드
                    response = requests.get(
                        f"{orthanc_url}/instances/{instance_id}/file",
                        auth=orthanc_auth,
                        timeout=30
                    )
                    response.raise_for_status()
                    
                    # Base64 인코딩 (내부 처리용)
                    slices_data.append(base64.b64encode(response.content).decode('utf-8'))
                    
                    if (slice_idx + 1) % 20 == 0:
                        logger.info(f"  시퀀스 {seq_idx+1}: {slice_idx+1}/{len(seq_instances)} 슬라이스 다운로드 완료")
                
                sequences_3d.append(slices_data)
                logger.info(f"✅ 시퀀스 {seq_idx+1}/4 다운로드 완료: {len(slices_data)}개 슬라이스")
            
            return {
                "sequences_3d": sequences_3d,  # base64 인코딩된 DICOM 배열
                "seg_series_uid": json_data.get("seg_series_uid"),
                "original_series_id": json_data.get("original_series_id"),
                "start_instance_number": json_data.get("start_instance_number", 1)
            }
        
        # 기존 방식 지원 (하위 호환성)
        if "sequences_3d" in json_data or "sequences" in json_data:
            logger.info("📥 기존 방식 입력 감지")
            return json_data
```
    
    def forward(self, data: dict) -> dict:
        """
        세그멘테이션 추론
        """
        # 1. 모델 로드 (최초 1회)
        if self.model is None:
            self.model = SwinUNETR(
                spatial_dims=3,
                in_channels=4,
                out_channels=1,
                feature_size=24
            )
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
        
        # 2. base64 디코딩 및 각 시퀀스를 3D 볼륨으로 변환
        sequences_3d = []
        original_dicom_ref = None
        
        # sequences_3d는 base64 인코딩된 DICOM 배열
        for seq_idx, seq_base64_list in enumerate(data['sequences_3d']):
            # 각 시퀀스의 base64 인코딩된 DICOM 파일들을 디코딩
            dicom_files = []
            for base64_dicom in seq_base64_list:
                # base64 디코딩 → DICOM 바이트
                dicom_bytes = base64.b64decode(base64_dicom)
                dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
                pixel_array = dicom.pixel_array.astype(np.float32)
                
                # 정규화
                if pixel_array.max() > pixel_array.min():
                    pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min())
                
                dicom_files.append(pixel_array)
                if original_dicom_ref is None:
                    original_dicom_ref = dicom
            
            # [D, H, W] 형태로 결합 (D=96, H=256, W=256)
            volume_3d = np.stack(dicom_files, axis=0)
            sequences_3d.append(volume_3d)
        
        # 3. 4개 시퀀스를 [4, 96, 96, 96]로 변환 (다운샘플링)
        from scipy.ndimage import zoom
        target_size = 96
        
        resized_sequences = []
        for seq_3d in sequences_3d:
            d, h, w = seq_3d.shape
            zoom_factors = (target_size/d, target_size/h, target_size/w)
            resized = zoom(seq_3d, zoom_factors, order=1)
            resized_sequences.append(resized)
        
        volume_4d = np.stack(resized_sequences, axis=0)  # [4, 96, 96, 96]
        
        # 4. 모델 추론
        with torch.no_grad():
            input_tensor = torch.from_numpy(volume_4d).float().unsqueeze(0).to(self.device)
            
            output = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=self.model,
                overlap=0.5
            )
            
            pred_prob = torch.sigmoid(output).cpu().numpy()[0, 0]  # [96, 96, 96]
        
        # 5. 후처리
        pred_mask = (pred_prob > 0.7).astype(np.uint8)  # 임계값 0.7
        
        # 형태학적 침식
        from scipy import ndimage
        mask_eroded = ndimage.binary_erosion(pred_mask, structure=np.ones((3,3,3)))
        
        # 구멍 채우기
        mask_filled = ndimage.binary_fill_holes(mask_eroded)
        
        # 작은 객체 제거
        labeled, num_features = ndimage.label(mask_filled)
        if num_features > 0:
            sizes = ndimage.sum(mask_filled, labeled, range(1, num_features + 1))
            max_label = np.argmax(sizes) + 1
            mask_cleaned = (labeled == max_label).astype(np.uint8)
        else:
            mask_cleaned = mask_filled.astype(np.uint8)
        
        # 원본 크기로 업샘플링 (현재는 제거됨)
        # mask_cleaned = zoom(mask_cleaned, zoom_factors_inv, order=0)
        # 최종 마스크: [96, 96, 96]
        
        # 6. DICOM SEG 생성
        seg_series_uid = generate_uid()
        seg_instance = create_dicom_seg_multiframe(
            original_dicom=original_dicom_ref,
            mask_array_3d=mask_cleaned,  # [96, H, W]
            seg_series_uid=seg_series_uid,
            start_instance_number=1,
            original_series_id=data['series_data'][0]['series_id']
        )
        
        # 7. Orthanc에 업로드
        orthanc_url = data.get('orthanc_url', ORTHANC_URL)
        orthanc_auth = data.get('orthanc_auth', (ORTHANC_USER, ORTHANC_PASSWORD))
        
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp:
            seg_instance.save_as(tmp.name)
            with open(tmp.name, 'rb') as f:
                seg_bytes = f.read()
            
            response = requests.post(
                f"{orthanc_url}/instances",
                auth=orthanc_auth,
                headers={'Content-Type': 'application/dicom'},
                data=seg_bytes,
                timeout=30
            )
            response.raise_for_status()
            seg_instance_id = response.json().get('ID')
        
        # 8. 결과 반환
        return {
            'seg_instance_id': seg_instance_id,
            'start_slice_index': 0,  # 첫 번째 슬라이스부터 시작
            'num_frames': 96
        }
    
    def serialize(self, data: dict) -> bytes:
        """결과 직렬화"""
        return json.dumps(data).encode('utf-8')
```

**DICOM SEG 구조**:
```python
def create_dicom_seg_multiframe(original_dicom, mask_array_3d, seg_series_uid, start_instance_number, original_series_id):
    """
    Multi-frame DICOM SEG 생성
    mask_array_3d: [96, H, W] 형태
    """
    num_frames = mask_array_3d.shape[0]  # 96
    
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # 환자 정보 (원본에서 복사)
    ds.PatientName = original_dicom.PatientName
    ds.PatientID = original_dicom.PatientID
    ds.StudyInstanceUID = original_dicom.StudyInstanceUID
    
    # 세그멘테이션 시리즈 정보
    ds.SeriesInstanceUID = seg_series_uid
    ds.SeriesNumber = '9999'
    ds.SeriesDescription = f'AI Tumor Segmentation (Original Series: {original_series_id})'
    ds.Modality = 'SEG'
    
    # Multi-frame 정보
    ds.NumberOfFrames = num_frames  # 96
    ds.Rows = mask_array_3d.shape[1]  # H
    ds.Columns = mask_array_3d.shape[2]  # W
    ds.InstanceNumber = '1'
    
    # 96개 프레임을 PixelData로 결합
    pixel_data_list = []
    for i in range(num_frames):
        frame_data = (mask_array_3d[i] * 255).astype(np.uint8)
        pixel_data_list.append(frame_data.tobytes())
    
    ds.PixelData = b''.join(pixel_data_list)
    
    return ds
```

---

### 10-11단계: Django - 응답 반환

**파일**: `backend/mri_viewer/segmentation_views.py`

```python
# Mosec 응답 처리 (이미 위에 포함됨)
result = response.json()
seg_instance_id = result.get('seg_instance_id')
start_slice_index = result.get('start_slice_index', 0)

return JsonResponse({
    'success': True,
    'seg_instance_id': seg_instance_id,
    'start_slice_index': start_slice_index,
    'num_frames': 96
})
```

**응답 형식**:
```json
{
  "success": true,
  "seg_instance_id": "abc123-def456-...",
  "start_slice_index": 0,
  "num_frames": 96
}
```

---

### 12-18단계: 프론트엔드 - 오버레이 표시

**파일**: `frontend/src/pages/MRIImageDetail.tsx`

```typescript
// 12. 세그멘테이션 결과 저장
const handleAiAnalysis = async () => {
  // ... AI 분석 실행
  
  const data = await response.json();
  
  if (data.success) {
    // 선택된 4개 시리즈 모두에 결과 저장
    const newResults: { [seriesId: string]: any } = {};
    selectedSeriesFor4Channel.forEach(seriesIndex => {
      const seriesId = seriesGroups[seriesIndex].series_id;
      newResults[seriesId] = {
        seg_instance_id: data.seg_instance_id,
        start_slice_index: data.start_slice_index
      };
    });
    
    setSeriesSegmentationResults(prev => ({ ...prev, ...newResults }));
    setShowSegmentationOverlay(true);
  }
};

// 13-14. 슬라이스 변경 시 프레임 로드
useEffect(() => {
  if (showSegmentationOverlay && currentSeriesId) {
    const result = seriesSegmentationResults[currentSeriesId];
    if (result) {
      loadSegmentationFrames(result.seg_instance_id, result.start_slice_index);
    }
  }
}, [selectedImageIndex, showSegmentationOverlay]);

const loadSegmentationFrames = async (segInstanceId: string, startIdx: number) => {
  // 슬라이스 인덱스 계산
  const frameIndex = selectedImageIndex - startIdx;
  
  if (frameIndex >= 0 && frameIndex < 96) {
    try {
      const response = await fetch(
        `/api/mri/segmentation/instances/${segInstanceId}/frames/?frame_index=${frameIndex}`
      );
      const data = await response.json();
      
      if (data.success) {
        // 프레임 데이터 저장
        setSegmentationFrames(prev => ({
          ...prev,
          [currentSeriesId]: {
            ...prev[currentSeriesId],
            [frameIndex]: data.frame_data
          }
        }));
      }
    } catch (error) {
      console.error('프레임 로드 실패:', error);
    }
  }
};

// 17-18. 오버레이 렌더링
const currentFrameData = segmentationFrames[currentSeriesId]?.[selectedImageIndex - startIdx];

return (
  <div className="relative">
    {/* 원본 MRI 이미지 */}
    <CornerstoneViewer
      imageId={currentImage?.preview_url}
      // ...
    />
    
    {/* 세그멘테이션 오버레이 */}
    {showSegmentationOverlay && currentFrameData && (
      <img
        src={`data:image/png;base64,${currentFrameData}`}
        alt="Segmentation Overlay"
        className="absolute inset-0 pointer-events-none"
        style={{
          transform: 'scaleX(-1)',  // 좌우 반전
          opacity: overlayOpacity,
          width: '100%',
          height: '100%'
        }}
      />
    )}
  </div>
);
```

---

### 14-16단계: Django - 프레임 추출

**파일**: `backend/mri_viewer/segmentation_views.py`

```python
@api_view(['GET'])
def get_segmentation_frames(request, seg_instance_id):
    """
    DICOM SEG에서 특정 프레임 추출
    """
    try:
        frame_index = int(request.GET.get('frame_index', 0))
        
        # Orthanc에서 SEG 인스턴스 다운로드
        orthanc_url = settings.ORTHANC_URL
        orthanc_auth = (settings.ORTHANC_USER, settings.ORTHANC_PASSWORD)
        client = Orthanc(orthanc_url, orthanc_auth)
        
        dicom_bytes = client.get_instance_file(seg_instance_id)
        dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
        
        # PixelData에서 프레임 추출
        num_frames = dicom.NumberOfFrames  # 96
        rows = dicom.Rows
        cols = dicom.Columns
        
        # 각 프레임의 크기 계산
        frame_size = rows * cols  # bytes (8-bit)
        
        # PixelData에서 frame_index 번째 프레임 추출
        start_byte = frame_index * frame_size
        end_byte = start_byte + frame_size
        frame_bytes = dicom.PixelData[start_byte:end_byte]
        
        # numpy 배열로 변환
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((rows, cols))
        
        # PNG로 변환 (base64)
        from PIL import Image
        import base64
        from io import BytesIO
        
        img = Image.fromarray(frame_array, mode='L')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return JsonResponse({
            'success': True,
            'frame_data': img_base64,
            'frame_index': frame_index
        })
        
    except Exception as e:
        logger.error(f"프레임 추출 실패: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
```

---

## 📊 데이터 크기 및 성능

### 각 단계별 데이터 크기

1. **프론트엔드 → Django**: ~1KB (JSON, series_ids만)
2. **Django → Mosec**: ~몇 KB (JSON, orthanc_instance_ids만) ✅
3. **Mosec → Orthanc (각 이미지)**: 
   - 요청: GET /instances/{id}/file
   - 응답: ~19MB (DICOM 파일 × 96슬라이스 × 4시퀀스 = ~7.3GB 총량)
   - ✅ Mosec이 Orthanc에서 직접 다운로드
4. **Mosec 내부 처리**: 
   - 다운로드 후 base64 디코딩 → DICOM 바이트
   - DICOM → 3D 볼륨 변환
   - 입력: [4, 96, 96, 96] float32 → ~14MB
   - 출력: [96, 96, 96] uint8 → ~3.5MB
5. **Mosec → Orthanc (SEG 업로드)**: ~3.5MB (DICOM SEG 파일)
6. **Django → 프론트엔드 (프레임)**: ~50-100KB (PNG base64, 프레임당)

### 처리 시간

- **Django → Mosec 요청**: ~10ms (instance_ids만 전송)
- **Mosec → Orthanc 다운로드**: ~2-5분 (4개 시퀀스 × 96슬라이스 = 384개 DICOM 파일)
  - 각 DICOM: ~19MB
  - 네트워크 속도에 따라 변동
- **base64 디코딩 및 3D 볼륨 변환**: ~1-2분
- **모델 추론**: ~30-60초 (96×96×96 sliding window)
- **DICOM SEG 생성 및 업로드**: ~2-3초
- **프레임 추출**: ~100-200ms (프레임당)

**총 처리 시간**: 약 5-10분 (DICOM 다운로드 시간 포함)

---

## 🔧 주요 설정 및 파라미터

### Mosec 설정
```bash
# /etc/systemd/system/mosec-segmentation.service
ExecStart=/usr/bin/python3 /home/shrjsdn908/segmentation_mosec.py \
  --port 5006 \
  --timeout 300000 \
  --max-batch-size 1
```

### 모델 파라미터
- **입력 크기**: [4, 96, 96, 96]
- **출력 크기**: [96, 96, 96]
- **ROI 크기**: (96, 96, 96)
- **Overlap**: 0.5
- **임계값**: 0.7

### DICOM SEG 파라미터
- **NumberOfFrames**: 96
- **InstanceNumber**: '1' (항상)
- **SeriesNumber**: '9999'
- **Modality**: 'SEG'
- **PixelRepresentation**: 0 (unsigned)
- **BitsAllocated**: 8

---

## 🔄 URL 라우팅

### Django URLs
```python
# backend/mri_viewer/urls.py
path('segmentation/series/<str:series_id>/segment/', segmentation_views.segment_series, name='segment-series'),
path('segmentation/instances/<str:seg_instance_id>/frames/', segmentation_views.get_segmentation_frames, name='get-segmentation-frames'),
```

### 프론트엔드 API 호출
```typescript
// 시리즈 세그멘테이션 (4-channel)
POST /api/mri/segmentation/series/{series_id}/segment/
Body: {
  "sequence_series_ids": [series1_id, series2_id, series3_id, series4_id]
}

// 프레임 추출
GET /api/mri/segmentation/instances/{seg_instance_id}/frames/
```

---

## ⚠️ 주요 주의사항

1. **아키텍처 개선 (신버전)**
   - ✅ Django → Mosec: instance_ids만 전송 (~몇 KB)
   - ✅ Mosec이 Orthanc에서 직접 DICOM 다운로드
   - ✅ 413 Request Entity Too Large 에러 방지
   - ✅ 네트워크 부하 감소

2. **메모리 사용량**
   - Mosec 내부에서 4개 시퀀스 × 96슬라이스 DICOM 로드
   - 약 7.3GB (384개 DICOM 파일 × ~19MB)
   - 디스크 임시 저장 권장 (스트리밍 처리)

3. **타임아웃 설정**
   - Django → Mosec: 2400초 (40분)
   - DICOM 다운로드 시간 고려 (2-5분)
   - 모델 추론 시간 고려 (30-60초)
   - Mosec → Orthanc 다운로드: 30초 (각 DICOM당)

3. **다운샘플링 문제**
   - 원본 256×256 → 모델 96×96
   - 정보 손실 발생
   - 재학습 권장 (256×256)

4. **좌우 반전**
   - DICOM 좌표계와 화면 좌표계 차이
   - `transform: scaleX(-1)` 필수

5. **슬라이스 인덱스 매핑**
   - `frameIndex = selectedImageIndex - startIdx`
   - 정확한 계산 필수

---

## 🔍 핵심 개선사항: 신버전 아키텍처

### 구버전 (base64 전송 방식)
```
Django → DICOM 다운로드 → base64 인코딩 → Mosec 전송 (~50-100MB)
❌ 413 Request Entity Too Large 에러 발생 가능
❌ 네트워크 부하 높음
```

### 신버전 (instance_ids 전송 방식) ✅
```
Django → instance_ids만 전송 (~몇 KB) → Mosec
Mosec → Orthanc에서 직접 DICOM 다운로드
✅ 413 에러 방지
✅ 네트워크 부하 감소
✅ 확장성 향상
```

### 비교표

| 항목 | 구버전 | 신버전 |
|------|--------|--------|
| **전송 데이터** | base64 DICOM (~50-100MB) | instance_ids만 (~몇 KB) |
| **다운로드 위치** | Django | Mosec (Orthanc API 직접 호출) |
| **413 에러** | 발생 가능 | 방지됨 |
| **네트워크 부하** | 높음 | 낮음 |
| **타임아웃** | 300초 | 2400초 (40분, 다운로드 시간 고려) |

---

**작성일**: 2026년 1월 10일 (신버전으로 업데이트)
**작성자**: AI Assistant

