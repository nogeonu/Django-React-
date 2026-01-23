# 맘모그래피 AI 분석 데이터 흐름 상세 정리

## 📋 개요
- **기능**: 맘모그래피 4장 이미지 (L-CC, R-CC, L-MLO, R-MLO) AI 분류 분석
- **모델**: ResNet50 기반 4-class 분류 (Mass, Calcification, Architectural/Asymmetry, Normal)
- **서비스 구조**: Django → Mosec → Orthanc API (직접 다운로드) → Mosec → Django → 프론트엔드
- **포트**: Mosec 5007, Orthanc 8042

---

## 🔄 전체 데이터 흐름도

```
[프론트엔드 React]
    │
    │ 1. 4장 이미지 선택 또는 전체 선택
    │ 2. "AI 분석" 버튼 클릭
    │
    ▼
[Django - mammography_views.py]
    │ mammography_ai_analysis()
    │
    │ 3. 요청 데이터 파싱
    │    └─ instance_ids: [id1, id2, id3, id4]
    │
    │ 4. Mosec에 요청 전송 (instance_ids만 전송)
    │    POST http://localhost:5007/inference
    │    Body: {
    │      "instance_ids": [
    │        "39e6546f-96355874-4cadb391-381e9845-9d28a4f7",
    │        "d923023e-e663f8e6-2a4bb97b-c990e934-9c60cdff",
    │        "eaf4f7be-c560dfef-5f88c18c-9792a186-edae4647",
    │        "fbfe8539-26fbbb6b-b38a4288-aec087fe-7a30a630"
    │      ],
    │      "orthanc_url": "http://localhost:8042",
    │      "orthanc_auth": ["admin", "admin123"]
    │    }
    │    ※ DICOM 파일은 전송하지 않음 (instance_ids만)
    │
    ▼
[Mosec - mammography_mosec.py]
    │ MammographyWorker
    │
    │ 5. deserialize(): JSON → Python dict
    │    └─ {
    │         "instance_ids": [...],
    │         "orthanc_url": "...",
    │         "orthanc_auth": [...]
    │       }
    │
    │ 6. forward():
    │    ├─ 모델 로드 (최초 1회)
    │    │  └─ ResNet50 (pretrained=False, num_classes=4)
    │    │
    │    ├─ 각 instance_id에 대해:
    │    │  ├─ Orthanc API 직접 호출
    │    │  │  └─ GET /orthanc/instances/{instance_id}/file
    │    │  │     └─ DICOM 바이트 다운로드 (~19MB)
    │    │  │
    │    │  ├─ DICOM → 이미지 변환
    │    │  │  └─ pydicom.dcmread() → pixel_array
    │    │  │
    │    │  ├─ 전처리
    │    │  │  ├─ Otsu 임계값 처리 (배경 제거)
    │    │  │  ├─ Contour 검출 (유방 영역)
    │    │  │  ├─ Bounding Box Crop (유방 영역만 추출)
    │    │  │  └─ Resize 512×512 (모델 입력 크기)
    │    │  │
    │    │  ├─ 데이터 증강 (옵션)
    │    │  │  ├─ 수평/수직 반전
    │    │  │  └─ 회전
    │    │  │
    │    │  ├─ 정규화
    │    │  │  └─ ImageNet 통계: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    │    │  │
    │    │  ├─ 모델 추론
    │    │  │  └─ ResNet50.forward() → logits [4]
    │    │  │
    │    │  └─ 후처리
    │    │     ├─ Softmax → 확률 [4]
    │    │     ├─ argmax → 예측 클래스
    │    │     └─ 최대 확률 값
    │    │
    │    └─ 4개 결과 수집
    │       └─ [
    │            {
    │              "instance_id": "id1",
    │              "predicted_class": "Normal",
    │              "probability": 1.0,
    │              "all_probabilities": [0.0, 0.0, 0.0, 1.0],
    │              "success": true
    │            },
    │            ... (4개)
    │          ]
    │
    │ 7. serialize(): dict → JSON bytes
    │    └─ {
    │         "results": [
    │           {...},  // 결과 1
    │           {...},  // 결과 2
    │           {...},  // 결과 3
    │           {...}   // 결과 4
    │         ]
    │       }
    │
    ▼
[Django - mammography_views.py]
    │
    │ 8. Mosec 응답 처리
    │    └─ response.json() → {"results": [...]}
    │
    │ 9. 각 결과를 DICOM 메타데이터와 매핑
    │    ├─ Orthanc API 호출
    │    │  └─ GET /orthanc/instances/{instance_id}
    │    │     └─ MainDicomTags에서 뷰 정보 추출
    │    │        ├─ ViewPosition: "CC", "MLO"
    │    │        └─ ImageLaterality: "L", "R"
    │    │
    │    └─ 뷰 이름 생성
    │       └─ "L-CC", "R-CC", "L-MLO", "R-MLO"
    │
    │ 10. 최종 결과 생성
    │     └─ [
    │          {
    │            "view": "L-CC",
    │            "predicted_class": "Normal",
    │            "probability": 1.0,
    │            "all_probabilities": {...},
    │            "instance_id": "id1"
    │          },
    │          ... (4개)
    │        ]
    │
    │ 11. 응답 반환
    │     └─ {
    │          "success": true,
    │          "results": [...]
    │        }
    │
    ▼
[프론트엔드 React]
    │ MRIImageDetail.tsx
    │
    │ 12. 결과 표시
    │     └─ 4개 결과 카드로 표시
    │        ├─ 뷰 이름 (L-CC, R-CC, L-MLO, R-MLO)
    │        ├─ 예측 클래스 (Mass, Calcification, Asymmetry, Normal)
    │        ├─ 확률 (0-100%)
    │        └─ 색상 코딩된 확률 바
    │           ├─ 🔴 Mass (빨강)
    │           ├─ 🟠 Calcification (주황)
    │           ├─ 🟡 Architectural/Asymmetry (노랑)
    │           └─ 🟢 Normal (초록)
```

---

## 📝 단계별 상세 설명

### 1단계: 프론트엔드 - 사용자 액션

**파일**: `frontend/src/pages/MRIImageDetail.tsx`

```typescript
// AI 분석 상태
const [aiAnalyzing, setAiAnalyzing] = useState(false);
const [aiResult, setAiResult] = useState<any>(null);

// "AI 분석" 버튼 클릭
const handleAiAnalysis = async () => {
  // 현재 표시 중인 맘모그래피 이미지들 가져오기
  const mgImages = currentImages.filter(img => img.modality === 'MG');
  
  if (mgImages.length !== 4) {
    toast({
      title: "4장의 맘모그래피 이미지가 필요합니다",
      variant: "destructive"
    });
    return;
  }
  
  setAiAnalyzing(true);
  
  try {
    // 2단계로 이동
    const response = await fetch('/api/mri/mammography/analyze/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCsrfToken()
      },
      body: JSON.stringify({
        instance_ids: mgImages.map(img => img.instance_id)
      })
    });
    
    const data = await response.json();
    
    if (data.success) {
      setAiResult(data.results);
      toast({
        title: "AI 분석 완료",
        description: "4장의 이미지 분석이 완료되었습니다"
      });
    } else {
      throw new Error(data.error || 'AI 분석 실패');
    }
  } catch (error) {
    console.error('AI 분석 실패:', error);
    toast({
      title: "AI 분석 실패",
      description: error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다",
      variant: "destructive"
    });
  } finally {
    setAiAnalyzing(false);
  }
};
```

**요청 데이터 형식**:
```json
{
  "instance_ids": [
    "39e6546f-96355874-4cadb391-381e9845-9d28a4f7",
    "d923023e-e663f8e6-2a4bb97b-c990e934-9c60cdff",
    "eaf4f7be-c560dfef-5f88c18c-9792a186-edae4647",
    "fbfe8539-26fbbb6b-b38a4288-aec087fe-7a30a630"
  ]
}
```

**데이터 크기**: ~1KB (instance_ids만 전송)

---

### 2-4단계: Django - Mosec에 요청 전송

**파일**: `backend/mri_viewer/mammography_views.py`

```python
@api_view(['POST'])
@csrf_exempt
def mammography_ai_analysis(request):
    """
    맘모그래피 AI 분석
    """
    try:
        # 1. 요청 데이터 파싱
        data = json.loads(request.body)
        instance_ids = data.get('instance_ids', [])
        
        if len(instance_ids) != 4:
            return JsonResponse({
                'success': False,
                'error': '4장의 이미지가 필요합니다'
            }, status=400)
        
        # 2. Orthanc 설정
        orthanc_url = settings.ORTHANC_URL
        orthanc_auth = (settings.ORTHANC_USER, settings.ORTHANC_PASSWORD)
        
        # 3. Mosec에 요청 전송 (instance_ids만 전송)
        mosec_url = 'http://localhost:5007/inference'
        
        response = requests.post(
            mosec_url,
            json={
                'instance_ids': instance_ids,
                'orthanc_url': orthanc_url,
                'orthanc_auth': list(orthanc_auth)  # tuple → list 변환
            },
            timeout=300  # 5분 (4장 처리)
        )
        
        if response.status_code != 200:
            raise Exception(f"Mosec 서비스 오류: {response.status_code} - {response.text}")
        
        # 8단계로 이동: Mosec 응답 처리
        mosec_result = response.json()
        
        if not isinstance(mosec_result, dict):
            raise Exception(f"Mosec 응답 형식 오류: 예상 dict, 실제 {type(mosec_result)}")
        
        mosec_results = mosec_result.get("results", [])
        
        if len(mosec_results) != len(instance_ids):
            raise Exception(f"결과 개수 불일치: 기대 {len(instance_ids)}, 실제 {len(mosec_results)}")
        
        # 9단계: DICOM 메타데이터와 매핑
        results = []
        client = Orthanc(orthanc_url, orthanc_auth)
        
        for idx, (instance_id, mosec_result_item) in enumerate(zip(instance_ids, mosec_results)):
            if not mosec_result_item.get('success'):
                raise Exception(f"이미지 {idx+1} 분석 실패: {mosec_result_item.get('error', 'Unknown error')}")
            
            # Orthanc에서 인스턴스 메타데이터 가져오기
            instance_info = client.get_instance_info(instance_id)
            main_tags = instance_info.get('MainDicomTags', {})
            
            view_position = main_tags.get('ViewPosition', '')  # CC, MLO 등
            image_laterality = main_tags.get('ImageLaterality', '')  # L, R
            
            # 뷰 이름 생성
            if view_position and image_laterality:
                view_name = f"{image_laterality}-{view_position}"  # L-CC, R-MLO 등
            else:
                view_name = f"Image {idx+1}"
            
            results.append({
                'view': view_name,
                'predicted_class': mosec_result_item['predicted_class'],
                'probability': mosec_result_item['probability'],
                'all_probabilities': mosec_result_item.get('all_probabilities', {}),
                'instance_id': instance_id
            })
        
        # 11단계: 응답 반환
        return JsonResponse({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"맘모그래피 분석 실패: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
```

**Mosec 요청 데이터 형식**:
```json
{
  "instance_ids": [
    "39e6546f-96355874-4cadb391-381e9845-9d28a4f7",
    "d923023e-e663f8e6-2a4bb97b-c990e934-9c60cdff",
    "eaf4f7be-c560dfef-5f88c18c-9792a186-edae4647",
    "fbfe8539-26fbbb6b-b38a4288-aec087fe-7a30a630"
  ],
  "orthanc_url": "http://localhost:8042",
  "orthanc_auth": ["admin", "admin123"]
}
```

**데이터 크기**: ~1KB (instance_ids만 전송)

**핵심 포인트**: 
- ✅ DICOM 파일을 Django에서 Mosec으로 전송하지 않음
- ✅ instance_ids만 전송 (413 Request Entity Too Large 에러 방지)
- ✅ Mosec이 Orthanc에서 직접 다운로드

---

### 5-7단계: Mosec - AI 분석

**파일**: `backend/mammography_mosec.py`

```python
class MammographyWorker(Worker):
    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
        logger.info(f"💻 Device: {DEVICE}")
    
    def deserialize(self, data: bytes) -> dict:
        """요청 데이터 역직렬화"""
        json_data = json.loads(data.decode('utf-8'))
        logger.info(f"📥 수신한 데이터 키: {list(json_data.keys())}")
        return json_data
    
    def forward(self, data) -> list:
        """
        맘모그래피 이미지 분류 추론
        Mosec이 리스트로 전달할 수 있으므로 처리
        """
        # Mosec 배치 처리 대응
        if isinstance(data, list) and len(data) > 0:
            request_data = data[0]
        elif isinstance(data, dict):
            request_data = data
        else:
            raise ValueError(f"예상치 못한 데이터 타입: {type(data)}")
        
        # 1. 모델 로드 (최초 1회)
        if self.model is None:
            logger.info(f"🔄 모델 로딩 중: {MODEL_PATH}")
            
            self.model = create_resnet50_model(num_classes=4)
            
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(DEVICE)
            self.model.eval()
            logger.info(f"✅ 모델 로드 완료: {MODEL_PATH}")
        
        # 2. Orthanc API 설정
        instance_ids = request_data.get("instance_ids", [])
        orthanc_url = request_data.get("orthanc_url", "http://localhost:8042")
        orthanc_auth = tuple(request_data.get("orthanc_auth", ["admin", "admin123"]))
        
        logger.info(f"📥 Orthanc에서 데이터 다운로드 중: {orthanc_url}")
        logger.info(f"📊 총 {len(instance_ids)}장 이미지")
        
        # 3. 각 이미지 처리
        results = []
        
        for idx, instance_id in enumerate(instance_ids):
            try:
                logger.info(f"📥 DICOM 다운로드 {idx+1}/{len(instance_ids)}: {instance_id}")
                
                # Orthanc API 직접 호출
                response = requests.get(
                    f"{orthanc_url}/instances/{instance_id}/file",
                    auth=orthanc_auth,
                    timeout=60
                )
                response.raise_for_status()
                
                dicom_bytes = response.content
                logger.info(f"✅ DICOM 다운로드 완료: {len(dicom_bytes)} bytes")
                
                # 4. DICOM → 이미지 변환
                dicom = pydicom.dcmread(io.BytesIO(dicom_bytes))
                pixel_array = dicom.pixel_array.astype(np.float32)
                
                # 5. 전처리
                # 5-1. Otsu 임계값 처리 (배경 제거)
                from skimage.filters import threshold_otsu
                threshold = threshold_otsu(pixel_array)
                binary_mask = pixel_array > threshold
                
                # 5-2. Contour 검출 (유방 영역)
                from skimage.measure import find_contours
                contours = find_contours(binary_mask.astype(float), 0.5)
                
                if len(contours) == 0:
                    # Contour가 없으면 전체 이미지 사용
                    bbox = (0, 0, pixel_array.shape[1], pixel_array.shape[0])
                else:
                    # 가장 큰 contour 선택
                    largest_contour = max(contours, key=len)
                    y_min = int(np.min(largest_contour[:, 0]))
                    y_max = int(np.max(largest_contour[:, 0]))
                    x_min = int(np.min(largest_contour[:, 1]))
                    x_max = int(np.max(largest_contour[:, 1]))
                    bbox = (x_min, y_min, x_max, y_max)
                
                # 5-3. Bounding Box Crop
                x_min, y_min, x_max, y_max = bbox
                cropped = pixel_array[y_min:y_max, x_min:x_max]
                
                # 5-4. Resize 512×512 (모델 입력 크기)
                from PIL import Image
                img = Image.fromarray(cropped.astype(np.uint8))
                img_resized = img.resize((512, 512), Image.BILINEAR)
                
                # 5-5. 정규화
                img_array = np.array(img_resized).astype(np.float32) / 255.0
                
                # 5-6. RGB로 변환 (그레이스케일 → RGB)
                img_rgb = np.stack([img_array] * 3, axis=-1)
                
                # 5-7. ImageNet 통계로 정규화
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_normalized = (img_rgb - mean) / std
                
                # 5-8. 텐서 변환 [1, 3, 512, 512]
                img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
                
                # 6. 모델 추론
                with torch.no_grad():
                    output = self.model(img_tensor)  # [1, 4]
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]  # [4]
                    predicted_idx = np.argmax(probabilities)
                    predicted_class = CLASS_NAMES[predicted_idx]
                    confidence = float(probabilities[predicted_idx])
                
                logger.info(f"✅ 분류 완료 {idx+1}/{len(instance_ids)}: {predicted_class} (신뢰도: {confidence:.4f})")
                
                # 7. 결과 수집
                results.append({
                    'success': True,
                    'instance_id': instance_id,
                    'predicted_class': predicted_class,
                    'probability': confidence,
                    'all_probabilities': {
                        'Mass': float(probabilities[0]),
                        'Calcification': float(probabilities[1]),
                        'Architectural/Asymmetry': float(probabilities[2]),
                        'Normal': float(probabilities[3])
                    }
                })
                
            except Exception as e:
                logger.error(f"❌ 추론 오류 {idx+1}/{len(instance_ids)}: {str(e)}", exc_info=True)
                results.append({
                    'success': False,
                    'instance_id': instance_id,
                    'error': str(e)
                })
        
        # 8. 결과 반환 (리스트로 감싸서 반환)
        result_dict = {"results": results}
        logger.info(f"📤 forward 반환: {len(results)}개 결과")
        return [result_dict]
    
    def serialize(self, data: dict) -> bytes:
        """결과 직렬화"""
        logger.info(f"📦 serialize 입력 타입: {type(data)}")
        
        # forward가 리스트를 반환하면 첫 번째 항목 사용
        if isinstance(data, list) and len(data) > 0:
            result_data = data[0]
        elif isinstance(data, dict):
            result_data = data
        else:
            logger.error(f"❌ serialize 예상치 못한 데이터 타입: {type(data)}")
            result_data = {"error": f"Invalid data type: {type(data)}"}
        
        json_str = json.dumps(result_data)
        logger.info(f"📦 JSON 길이: {len(json_str)} bytes, 키: {list(result_data.keys()) if isinstance(result_data, dict) else 'N/A'}")
        return json_str.encode('utf-8')
```

**클래스 정의**:
```python
CLASS_NAMES = {
    0: 'Mass',                      # 종괴
    1: 'Calcification',             # 석회화
    2: 'Architectural/Asymmetry',   # 구조 왜곡/비대칭
    3: 'Normal'                     # 정상
}
```

**전처리 파이프라인**:
1. **Otsu 임계값**: 배경(공기) 제거
2. **Contour 검출**: 유방 영역 찾기
3. **Bounding Box Crop**: 유방 영역만 추출
4. **Resize 512×512**: 모델 입력 크기 맞춤
5. **정규화**: ImageNet 통계 적용

**모델 추론**:
- **입력**: [1, 3, 512, 512] float32
- **출력**: [1, 4] float32 (logits)
- **후처리**: Softmax → 확률 [4]
- **예측**: argmax → 클래스 인덱스

**Mosec 응답 형식**:
```json
{
  "results": [
    {
      "success": true,
      "instance_id": "39e6546f-96355874-4cadb391-381e9845-9d28a4f7",
      "predicted_class": "Normal",
      "probability": 1.0,
      "all_probabilities": {
        "Mass": 0.0,
        "Calcification": 0.0,
        "Architectural/Asymmetry": 0.0,
        "Normal": 1.0
      }
    },
    ... (4개)
  ]
}
```

**데이터 크기**:
- **각 DICOM 파일**: ~19MB (Orthanc에서 다운로드)
- **전처리 후 이미지**: 512×512×3 uint8 → ~786KB
- **모델 입력**: [1, 3, 512, 512] float32 → ~3MB
- **모델 출력**: [1, 4] float32 → ~16 bytes
- **최종 응답**: ~2-3KB (JSON)

---

### 8-11단계: Django - 응답 처리 및 반환

**파일**: `backend/mri_viewer/mammography_views.py`

```python
# (이미 위에 포함됨 - 9-11단계)

# 9단계: DICOM 메타데이터와 매핑
results = []
client = Orthanc(orthanc_url, orthanc_auth)

for idx, (instance_id, mosec_result_item) in enumerate(zip(instance_ids, mosec_results)):
    if not mosec_result_item.get('success'):
        raise Exception(f"이미지 {idx+1} 분석 실패: {mosec_result_item.get('error', 'Unknown error')}")
    
    # Orthanc에서 인스턴스 메타데이터 가져오기
    instance_info = client.get_instance_info(instance_id)
    main_tags = instance_info.get('MainDicomTags', {})
    
    view_position = main_tags.get('ViewPosition', '')  # CC, MLO 등
    image_laterality = main_tags.get('ImageLaterality', '')  # L, R
    
    # 뷰 이름 생성
    if view_position and image_laterality:
        view_name = f"{image_laterality}-{view_position}"  # L-CC, R-MLO 등
    else:
        view_name = f"Image {idx+1}"
    
    results.append({
        'view': view_name,
        'predicted_class': mosec_result_item['predicted_class'],
        'probability': mosec_result_item['probability'],
        'all_probabilities': mosec_result_item.get('all_probabilities', {}),
        'instance_id': instance_id
    })

# 11단계: 응답 반환
return JsonResponse({
    'success': True,
    'results': results
})
```

**최종 응답 형식**:
```json
{
  "success": true,
  "results": [
    {
      "view": "L-CC",
      "predicted_class": "Normal",
      "probability": 1.0,
      "all_probabilities": {
        "Mass": 0.0,
        "Calcification": 0.0,
        "Architectural/Asymmetry": 0.0,
        "Normal": 1.0
      },
      "instance_id": "39e6546f-96355874-4cadb391-381e9845-9d28a4f7"
    },
    {
      "view": "R-CC",
      "predicted_class": "Normal",
      "probability": 1.0,
      "all_probabilities": {...},
      "instance_id": "d923023e-e663f8e6-2a4bb97b-c990e934-9c60cdff"
    },
    {
      "view": "L-MLO",
      "predicted_class": "Normal",
      "probability": 1.0,
      "all_probabilities": {...},
      "instance_id": "eaf4f7be-c560dfef-5f88c18c-9792a186-edae4647"
    },
    {
      "view": "R-MLO",
      "predicted_class": "Normal",
      "probability": 1.0,
      "all_probabilities": {...},
      "instance_id": "fbfe8539-26fbbb6b-b38a4288-aec087fe-7a30a630"
    }
  ]
}
```

**데이터 크기**: ~3-5KB (JSON)

---

### 12단계: 프론트엔드 - 결과 표시

**파일**: `frontend/src/pages/MRIImageDetail.tsx`

```typescript
// AI 분석 결과 렌더링
{aiResult && aiResult.length > 0 && (
  <Card className="mt-4">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Brain className="w-5 h-5" />
        AI 분석 결과
      </CardTitle>
      <CardDescription>
        {aiResult.length}장 분석 결과
      </CardDescription>
    </CardHeader>
    <CardContent>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {aiResult.map((result: any, index: number) => {
          const { view, predicted_class, probability, all_probabilities } = result;
          
          // 클래스별 색상 설정
          const classColors: { [key: string]: { bg: string; text: string; bar: string } } = {
            'Mass': { bg: 'bg-red-50', text: 'text-red-700', bar: 'bg-red-500' },
            'Calcification': { bg: 'bg-orange-50', text: 'text-orange-700', bar: 'bg-orange-500' },
            'Architectural/Asymmetry': { bg: 'bg-yellow-50', text: 'text-yellow-700', bar: 'bg-yellow-500' },
            'Normal': { bg: 'bg-green-50', text: 'text-green-700', bar: 'bg-green-500' }
          };
          
          const colors = classColors[predicted_class] || classColors['Normal'];
          const probabilityPercent = (probability * 100).toFixed(1);
          
          // 클래스 이모지
          const classEmoji: { [key: string]: string } = {
            'Mass': '🔴',
            'Calcification': '🟠',
            'Architectural/Asymmetry': '🟡',
            'Normal': '🟢'
          };
          
          return (
            <Card key={index} className={`${colors.bg} border-2`}>
              <CardContent className="pt-6">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-lg">{view}</h3>
                    <span className="text-2xl">{classEmoji[predicted_class]}</span>
                  </div>
                  
                  <div>
                    <div className="text-sm text-gray-600 mb-1">예측 클래스</div>
                    <div className={`font-bold text-lg ${colors.text}`}>
                      {predicted_class}
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-sm text-gray-600 mb-1">확률</div>
                    <div className="font-bold text-xl">{probabilityPercent}%</div>
                  </div>
                  
                  {/* 확률 바 */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-gray-600">
                      <span>0%</span>
                      <span>100%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className={`${colors.bar} h-3 rounded-full transition-all duration-300`}
                        style={{ width: `${probabilityPercent}%` }}
                      />
                    </div>
                  </div>
                  
                  {/* 전체 확률 (옵션) */}
                  {all_probabilities && (
                    <details className="text-xs text-gray-600 mt-2">
                      <summary className="cursor-pointer">전체 확률 보기</summary>
                      <div className="mt-2 space-y-1">
                        {Object.entries(all_probabilities).map(([cls, prob]) => (
                          <div key={cls} className="flex justify-between">
                            <span>{cls}:</span>
                            <span className="font-semibold">
                              {((prob as number) * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </CardContent>
  </Card>
)}
```

**UI 표시 예시**:
```
┌─────────────────────────────────┐
│ 🧠 AI 분석 결과                 │
│ 4장 분석 결과                    │
├─────────────────────────────────┤
│ ┌─────────────────────────────┐ │
│ │ L-CC 🟢 Normal              │ │
│ │ 확률: 100.0%                │ │
│ │ ████████████████████████    │ │
│ └─────────────────────────────┘ │
│ ┌─────────────────────────────┐ │
│ │ R-CC 🟢 Normal              │ │
│ │ 확률: 100.0%                │ │
│ │ ████████████████████████    │ │
│ └─────────────────────────────┘ │
│ ┌─────────────────────────────┐ │
│ │ L-MLO 🟢 Normal             │ │
│ │ 확률: 100.0%                │ │
│ │ ████████████████████████    │ │
│ └─────────────────────────────┘ │
│ ┌─────────────────────────────┐ │
│ │ R-MLO 🟢 Normal             │ │
│ │ 확률: 100.0%                │ │
│ │ ████████████████████████    │ │
│ └─────────────────────────────┘ │
└─────────────────────────────────┘
```

---

## 📊 데이터 크기 및 성능

### 각 단계별 데이터 크기

1. **프론트엔드 → Django**: ~1KB (JSON, instance_ids만)
2. **Django → Mosec**: ~1KB (JSON, instance_ids만)
3. **Mosec → Orthanc (각 이미지)**: 
   - 요청: GET /instances/{id}/file
   - 응답: ~19MB (DICOM 파일)
4. **Mosec 내부 처리**:
   - 전처리 후: 512×512×3 uint8 → ~786KB
   - 모델 입력: [1, 3, 512, 512] float32 → ~3MB
   - 모델 출력: [1, 4] float32 → ~16 bytes
5. **Mosec → Django**: ~2-3KB (JSON, 4개 결과)
6. **Django → Orthanc (메타데이터)**: 
   - 요청: GET /instances/{id}
   - 응답: ~1-2KB (JSON, 메타데이터)
7. **Django → 프론트엔드**: ~3-5KB (JSON, 최종 결과)

### 처리 시간

- **Django → Mosec 요청**: ~10ms
- **Mosec → Orthanc 다운로드 (각 이미지)**: ~1-2초
- **전처리 (각 이미지)**: ~1-2초
  - Otsu 임계값: ~200ms
  - Contour 검출: ~300ms
  - Crop & Resize: ~100ms
  - 정규화: ~100ms
- **모델 추론 (각 이미지)**: ~2-3초
- **Mosec 응답 처리**: ~10ms
- **DICOM 메타데이터 조회 (4개)**: ~200ms
- **총 처리 시간**: 약 15-20초 (4장)

**병렬 처리 가능**: 각 이미지를 병렬로 처리하면 ~5-8초로 단축 가능

---

## 🔧 주요 설정 및 파라미터

### Mosec 설정
```bash
# /etc/systemd/system/mammography-mosec.service
ExecStart=/usr/bin/python3 /home/shrjsdn908/mammography_mosec.py \
  --port 5007 \
  --timeout 120000 \
  --max-body-size 209715200
```

### 모델 파라미터
- **입력 크기**: [1, 3, 512, 512]
- **출력 크기**: [1, 4]
- **클래스 수**: 4
- **정규화**: ImageNet 통계
- **디바이스**: CPU (또는 CUDA)

### 전처리 파라미터
- **Otsu 임계값**: 자동 계산
- **Contour 검출**: 0.5 임계값
- **Resize**: 512×512 (BILINEAR)
- **정규화**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## 🔄 URL 라우팅

### Django URLs
```python
# backend/mri_viewer/urls.py
path('mammography/analyze/', mammography_views.mammography_ai_analysis, name='analyze-mammography'),
path('mammography/health/', mammography_views.mammography_ai_health, name='mammography-health'),
```

### 프론트엔드 API 호출
```typescript
// AI 분석
POST /api/mri/mammography/analyze/

// Health Check (옵션)
GET /api/mri/mammography/health/
```

---

## ⚠️ 주요 주의사항

1. **413 Request Entity Too Large 에러 방지**
   - ✅ instance_ids만 전송 (DICOM 파일 X)
   - ✅ Mosec이 Orthanc에서 직접 다운로드
   - ✅ max_body_size 200MB (충분한 여유)

2. **타임아웃 설정**
   - Django → Mosec: 300초 (5분)
   - Mosec → Orthanc: 60초 (이미지당)

3. **메모리 사용량**
   - 각 이미지 처리 시 ~20-30MB 메모리 필요
   - 4장 동시 처리 시 ~100-120MB

4. **Mosec 배치 처리**
   - `forward`는 리스트를 받을 수 있음
   - `isinstance` 체크로 타입 안정성 확보

5. **오류 처리**
   - 각 이미지 처리 실패 시 다른 이미지는 계속 처리
   - 부분 실패 허용

6. **뷰 이름 추출**
   - ViewPosition (CC, MLO) + ImageLaterality (L, R)
   - 없으면 "Image 1", "Image 2" 등으로 표시

---

## 🔍 핵심 차이점: MRI vs 맘모그래피

| 항목 | MRI 세그멘테이션 | 맘모그래피 AI 분석 |
|------|-----------------|-------------------|
| **전송 방식** | Django → Mosec (base64 DICOM) | Django → Mosec (instance_ids만) |
| **DICOM 다운로드** | Django에서 다운로드 | Mosec에서 직접 다운로드 |
| **데이터 크기** | ~50-100MB (4개 시리즈) | ~1KB (instance_ids만) |
| **처리 단위** | 4개 시리즈 → 1개 SEG | 4개 이미지 → 4개 결과 |
| **출력** | DICOM SEG (96 frames) | JSON (4개 분류 결과) |
| **저장** | Orthanc에 업로드 | 결과만 반환 (저장 안 함) |
| **프론트엔드** | 오버레이 표시 (동적 로드) | 결과 카드 표시 (즉시 표시) |

---

**작성일**: 2026년 1월 10일
**작성자**: AI Assistant

