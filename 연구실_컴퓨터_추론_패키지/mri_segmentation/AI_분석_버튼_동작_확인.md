# AI 분석 버튼 동작 확인

## 🔍 현재 상황

**프론트엔드에서 "AI 분석" 버튼을 누르면:**

1. ✅ 프론트엔드가 `/api/mri/segmentation/series/{series_id}/segment/` API 호출
2. ✅ Django 백엔드의 `segment_series()` 함수 실행
3. ⚠️ **환경 변수 `USE_LOCAL_INFERENCE` 확인**
   - `true`이면 → 연구실 컴퓨터 워커 사용 ✅
   - `false`이면 → GCP 서버에서 직접 추론 실행 ❌

---

## ✅ 연구실 컴퓨터에서 추론되려면

### 1. GCP Django 서버 환경 변수 설정 (필수!)

**GCP 서버에서 다음 명령어 실행:**

```bash
# 환경 변수 설정
export USE_LOCAL_INFERENCE=true

# 또는 .env 파일에 추가
echo "USE_LOCAL_INFERENCE=true" >> /srv/django-react/app/.env

# Gunicorn 재시작
sudo systemctl restart gunicorn
```

**또는 systemd 서비스 파일에 추가:**

```ini
[Service]
Environment="USE_LOCAL_INFERENCE=true"
```

### 2. 연구실 컴퓨터 워커 실행 (필수!)

**연구실 컴퓨터에서:**

```bash
# Windows
start_worker_http.bat

# 또는 Linux/Mac
python local_inference_worker_http.py
```

---

## 🔄 전체 동작 흐름

```
[프론트엔드] "AI 분석" 버튼 클릭
    ↓
[프론트엔드] POST /api/mri/segmentation/series/{series_id}/segment/
    Body: { "sequence_series_ids": [id1, id2, id3, id4] }
    ↓
[Django] segment_series() 함수
    ↓
[Django] USE_LOCAL_INFERENCE 환경 변수 확인
    ↓
[조건 분기]
    ├─ USE_LOCAL_INFERENCE=true
    │   ↓
    │   [Django] request_local_inference() 호출
    │   ↓
    │   [Django] 요청 파일 생성 (/tmp/mri_inference_requests/*.json)
    │   ↓
    │   [연구실 컴퓨터 워커] HTTP API 폴링:
    │       GET /api/mri/segmentation/pending-requests/
    │   ↓
    │   [연구실 컴퓨터 워커] 요청 발견 → 추론 실행
    │   ↓
    │   [연구실 컴퓨터 워커] 결과 업로드
    │   ↓
    │   [Django] 결과 반환 → 프론트엔드 표시 ✅
    │
    └─ USE_LOCAL_INFERENCE=false (또는 없음)
        ↓
        [Django] GCP 서버에서 직접 추론 실행
        ↓
        [Django] 결과 반환 → 프론트엔드 표시 ❌
```

---

## ⚠️ 현재 상태 확인 방법

### 1. GCP 서버 환경 변수 확인

```bash
# GCP 서버에서
echo $USE_LOCAL_INFERENCE

# 또는 Django에서 확인
python manage.py shell
>>> import os
>>> os.getenv('USE_LOCAL_INFERENCE')
```

### 2. Django 로그 확인

**"AI 분석" 버튼 클릭 시 로그에서 확인:**

```
# 연구실 컴퓨터 사용 시:
🏠 연구실 컴퓨터 워커를 통해 추론 요청 생성

# GCP 서버 사용 시:
☁️ GCP 서버에서 직접 추론 실행
```

### 3. 연구실 컴퓨터 워커 실행 확인

```bash
# Windows
tasklist | findstr python

# Linux/Mac
ps aux | grep local_inference_worker
```

---

## 🎯 즉시 적용 방법

### 방법 1: 환경 변수 설정 (권장)

**GCP 서버에서:**

```bash
# 1. 환경 변수 설정
sudo nano /etc/systemd/system/gunicorn.service

# 2. [Service] 섹션에 추가:
Environment="USE_LOCAL_INFERENCE=true"

# 3. 재로드 및 재시작
sudo systemctl daemon-reload
sudo systemctl restart gunicorn

# 4. 확인
sudo systemctl status gunicorn
```

### 방법 2: 프론트엔드에서 파라미터 추가 (임시)

**프론트엔드 코드 수정:**

```typescript
// MRIImageDetail.tsx 또는 MRIViewer.tsx
const response = await fetch(
  `/api/mri/segmentation/series/${seriesId}/segment/?use_local=true`,
  {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  }
);
```

**하지만 방법 1이 더 깔끔합니다!**

---

## ✅ 체크리스트

연구실 컴퓨터에서 추론되려면:

- [ ] GCP 서버에 `USE_LOCAL_INFERENCE=true` 환경 변수 설정
- [ ] Gunicorn 재시작 완료
- [ ] 연구실 컴퓨터에서 워커 실행 중
- [ ] 연구실 컴퓨터가 GCP 서버에 접근 가능 (인터넷 연결)
- [ ] 프론트엔드에서 "AI 분석" 버튼 클릭 테스트

---

## 🔧 문제 해결

### Q1: 여전히 GCP에서 실행됨

**원인**: 환경 변수가 설정되지 않았거나 Gunicorn이 재시작되지 않음

**해결**:
```bash
# 환경 변수 확인
sudo systemctl show gunicorn | grep USE_LOCAL_INFERENCE

# 재시작
sudo systemctl restart gunicorn

# 로그 확인
sudo journalctl -u gunicorn -f
```

### Q2: 워커가 요청을 받지 못함

**원인**: 워커가 실행되지 않았거나 Django API에 접근 불가

**해결**:
```bash
# 워커 실행 확인
ps aux | grep local_inference_worker

# Django API 테스트
curl http://34.42.223.43/api/mri/segmentation/pending-requests/
```

---

**작성일**: 2026년 1월
