# 맘모그래피 AI 분석 구현 오류 해결 정리

## 📋 개요
- **기능**: 맘모그래피 4장 이미지 (L-CC, R-CC, L-MLO, R-MLO) AI 분류 분석
- **모델**: ResNet50 기반 4-class 분류 (Mass, Calcification, Architectural/Asymmetry, Normal)
- **서비스**: Mosec (포트 5007)
- **기간**: 2026년 1월 10일

---

## 🔴 오류 1: 413 Request Entity Too Large

### 증상
```
POST http://34.42.223.43/api/mri/mammography/analyze/ 413 (Request Entity Too Large)
Mosec 서비스 오류: 413 - request body is too large
```

### 원인
- DICOM 파일 크기가 약 **19MB씩** (4장 = 약 76MB)
- Django에서 각 DICOM을 base64 인코딩하여 JSON으로 전송
- base64 인코딩으로 크기 증가 (19MB → 약 25MB)
- Mosec의 기본 `max_body_size` 제한 초과

### 해결 과정
1. **1차 시도**: `max_body_size`를 50MB로 증가 → 실패 (여전히 부족)
2. **2차 시도**: 100MB로 증가 → 실패
3. **3차 시도**: 200MB로 증가 → 성공
4. **최종 해결**: Mosec이 Orthanc에서 직접 DICOM을 다운로드하도록 아키텍처 변경

### 최종 해결 방법
**아키텍처 변경**: 대용량 DICOM 파일을 네트워크로 전송하지 않고, Mosec이 Orthanc API를 직접 호출

```python
# 변경 전 (에러)
Django → base64 인코딩 → JSON → Mosec (413 에러)

# 변경 후 (성공) ✅
Django → instance_ids, orthanc_url, orthanc_auth → Mosec
Mosec → Orthanc API 직접 호출 → DICOM 다운로드
```

### 수정 파일
- `backend/mammography_mosec.py`: `forward` 함수 수정
- `backend/mri_viewer/mammography_views.py`: 요청 데이터 구조 변경
- `/etc/systemd/system/mammography-mosec.service`: `max_body_size` 200MB

---

## 🔴 오류 2: Connection Refused

### 증상
```
Connection refused
Mosec 서비스 오류: Connection refused
```

### 원인
- `max_body_size` 변경 후 Mosec 서비스가 완전히 재시작되지 않음
- 또는 서비스가 시작 중인 상태

### 해결
```bash
sudo systemctl restart mammography-mosec
sudo systemctl status mammography-mosec
```

---

## 🔴 오류 3: AttributeError - 'list' object has no attribute 'get'

### 증상
```
AttributeError: 'list' object has no attribute 'get'
Mosec 서비스 오류: AttributeError
```

### 원인
- Mosec이 `forward` 함수에 **리스트**를 전달할 수 있음
- 코드는 **딕셔너리**만 가정하고 작성됨
- Mosec의 배치 처리 방식으로 리스트 전달 가능

### 해결
`forward` 함수에서 리스트/딕셔너리 둘 다 처리하도록 수정:

```python
# 수정 전 (에러)
def forward(self, data: dict) -> dict:
    instance_ids = data.get("instance_ids", [])  # ❌ 리스트면 에러

# 수정 후 (성공) ✅
def forward(self, data) -> dict:
    # 리스트/딕셔너리 둘 다 처리
    if isinstance(data, list) and len(data) > 0:
        request_data = data[0]
    elif isinstance(data, dict):
        request_data = data
    else:
        raise ValueError(f"예상치 못한 데이터 타입: {type(data)}")
    
    instance_ids = request_data.get("instance_ids", [])  # ✅
```

### 수정 파일
- `backend/mammography_mosec.py`: `forward` 함수 타입 체크 추가

---

## 🔴 오류 4: Mosec 응답 형식 오류 - 예상 dict, 실제 <class 'str'>

### 증상
```
Mosec 응답 형식 오류: 예상 dict, 실제 <class 'str'>
❌ 실제 응답: "results"
```

### 원인
- Mosec이 `forward`의 딕셔너리 반환값 `{"results": [...]}`을 **반복**하면서 **키 "results"**를 `serialize`에 전달
- `serialize` 함수가 문자열을 받아서 에러 발생

### 해결 시도 (1차 - 실패)
`serialize` 함수에서 타입 체크 추가:
```python
def serialize(self, data) -> bytes:
    if isinstance(data, list) and len(data) > 0:
        result_data = data[0]
    elif isinstance(data, dict):
        result_data = data
    # ...
```
→ 여전히 문자열 "results"가 전달됨

### 최종 해결
**`forward`가 리스트를 반환**하도록 변경:

```python
# 수정 전 (에러)
def forward(self, data) -> dict:
    return {"results": results}  # ❌ Mosec이 키를 반복

# 수정 후 (성공) ✅
def forward(self, data) -> list:
    result_dict = {"results": results}
    return [result_dict]  # ✅ 리스트로 감싸서 반환
```

### 수정 파일
- `backend/mammography_mosec.py`: 
  - `forward` 반환 타입: `dict` → `list`
  - `serialize` 함수 단순화 (딕셔너리만 처리)

---

## 🔴 오류 5: 결과 개수 불일치 - 기대 4, 실제 0

### 증상
```
결과 개수 불일치: 기대 4, 실제 0
Mosec 응답 처리 실패: results가 리스트가 아님
```

### 원인
- 오류 4를 해결하는 과정에서 `serialize`가 잘못된 데이터를 반환
- `serialize`에 문자열이 전달되면서 에러 딕셔너리를 반환
- Django에서 `results` 키가 없거나 빈 배열

### 해결
오류 4의 최종 해결 방법으로 해결됨:
- `forward`가 `[{"results": [...]}]` 리스트 반환
- `serialize`가 딕셔너리를 정상적으로 JSON 직렬화
- Django가 첫 번째 항목에서 `results` 추출

### 수정 파일
- `backend/mammography_mosec.py`: 최종 구조 확정
- `backend/mri_viewer/mammography_views.py`: 디버깅 로그 추가

---

## ✅ 최종 해결된 구조

### 데이터 흐름
```
1. 프론트엔드 (React)
   └─ POST /api/mri/mammography/analyze/
      └─ body: { "instance_ids": [id1, id2, id3, id4] }

2. Django (mammography_views.py)
   └─ Orthanc URL, 인증 정보 추가
   └─ POST http://localhost:5007/inference
      └─ body: {
           "instance_ids": [id1, id2, id3, id4],
           "orthanc_url": "http://localhost:8042",
           "orthanc_auth": ["admin", "admin123"]
         }

3. Mosec (mammography_mosec.py)
   └─ deserialize: JSON → dict
   └─ forward: 
      ├─ Orthanc에서 각 DICOM 다운로드
      ├─ 전처리 (Otsu, contour, crop, resize 512x512)
      ├─ ResNet50 모델 추론 (4-class)
      └─ return [{"results": [결과1, 결과2, 결과3, 결과4]}]
   └─ serialize: dict → JSON bytes
      └─ 각 딕셔너리를 JSON으로 직렬화

4. Django (응답 처리)
   └─ response.json() → {"results": [...]}
   └─ 각 결과를 DICOM 태그와 매핑 (뷰 정보: L-CC, R-MLO 등)

5. 프론트엔드
   └─ 4개 결과를 카드 형식으로 표시
      ├─ 뷰 이름 (L-CC, R-CC, L-MLO, R-MLO)
      ├─ 예측 클래스 (Mass, Calcification, Asymmetry, Normal)
      ├─ 확률 (0-100%)
      └─ 색상 코딩된 확률 바
```

### 핵심 포인트

1. **대용량 파일 전송 문제 해결**
   - ❌ Django → Mosec: base64 인코딩된 DICOM 전송 (76MB+)
   - ✅ Django → Mosec: instance_ids만 전송 (수 KB)
   - ✅ Mosec → Orthanc: 직접 DICOM 다운로드

2. **Mosec 배치 처리 이해**
   - Mosec은 `forward` 반환값을 **리스트**로 처리
   - `forward`가 딕셔너리를 반환하면 키를 반복할 수 있음
   - 해결: `forward`가 `list[dict]` 반환

3. **타입 안정성**
   - `forward`와 `serialize` 모두 다양한 타입 처리 가능하도록 작성
   - `isinstance` 체크로 런타임 타입 검증

---

## 📊 최종 테스트 결과

### 성공 케이스
- ✅ 4장 이미지 모두 정상 분석
- ✅ 결과: 모두 Normal (신뢰도 100%)
- ✅ 응답 시간: 약 15초 (4장 처리)

### 로그 확인
```bash
# Mosec 로그
sudo journalctl -u mammography-mosec -f

# Django 로그
sudo journalctl -u gunicorn -f
```

---

## 🔧 설정 파일

### `/etc/systemd/system/mammography-mosec.service`
```ini
[Unit]
Description=Mammography Mosec Service
After=network.target

[Service]
Type=simple
User=shrjsdn908
WorkingDirectory=/home/shrjsdn908
ExecStart=/usr/bin/python3 /home/shrjsdn908/mammography_mosec.py --port 5007 --timeout 120000 --max-body-size 209715200
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 주요 파라미터
- `--port 5007`: Mosec 서비스 포트
- `--timeout 120000`: 타임아웃 120초 (4장 처리 시간 고려)
- `--max-body-size 209715200`: 최대 요청 크기 200MB

---

## 📝 교훈

1. **아키텍처 설계의 중요성**
   - 대용량 데이터는 직접 전송보다 ID만 전송하고 서비스에서 직접 다운로드

2. **프레임워크 동작 방식 이해**
   - Mosec의 배치 처리 방식을 정확히 이해하고 맞춰서 구현

3. **타입 체크의 중요성**
   - 런타임 타입 검증으로 예상치 못한 입력 처리

4. **디버깅 로그의 활용**
   - 각 단계별 상세 로그로 문제 빠르게 파악

5. **점진적 해결**
   - 작은 문제부터 하나씩 해결하면서 근본 원인 파악

---

## 🎯 최종 상태

✅ **모든 오류 해결 완료**
✅ **4장 이미지 AI 분석 정상 작동**
✅ **프론트엔드 UI 정상 표시**
✅ **결과 정확도 검증 필요** (병변 이미지로 테스트 권장)

---

**작성일**: 2026년 1월 10일
**작성자**: AI Assistant
**검증자**: 사용자 (실제 테스트 완료)

