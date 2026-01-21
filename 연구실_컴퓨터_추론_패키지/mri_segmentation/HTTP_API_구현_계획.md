# HTTP API 방식 구현 계획

**작성일**: 2026년 1월 20일  
**통신 방식**: HTTP API (폴링)

---

## 🎯 전체 구조

```
[프론트엔드] "AI 분석" 버튼 클릭
    ↓
[GCP Django] DB에 요청 저장 (status='pending')
    ↓
[연구실 PC 워커] 30초마다 HTTP GET 요청
    ← GET /api/inference/pending
[GCP Django] 대기 중인 요청 반환
    ↓
[연구실 PC 워커] 추론 실행
    ↓
[연구실 PC 워커] 결과 전송
    → POST /api/inference/{id}/complete
[GCP Django] 결과 저장 및 프론트엔드 응답
```

---

## 👥 역할 분담

### 🔵 조원님 (GCP Django 서버)

#### 1. Django 모델 생성

```python
# models.py
from django.db import models
from django.utils import timezone

class InferenceRequest(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    # 요청 정보
    series_id = models.CharField(max_length=255, unique=True)
    series_ids = models.JSONField()  # 4개 시리즈 ID 리스트
    
    # 상태
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # 결과
    result = models.JSONField(null=True, blank=True)
    seg_instance_id = models.CharField(max_length=255, null=True, blank=True)
    
    # 시간
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
```

#### 2. API 엔드포인트 생성

```python
# views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.utils import timezone
from .models import InferenceRequest

@api_view(['GET'])
def get_pending_inference(request):
    """
    대기 중인 추론 요청 반환
    연구실 PC 워커가 30초마다 호출
    """
    # 가장 오래된 pending 요청 가져오기
    pending = InferenceRequest.objects.filter(status='pending').first()
    
    if pending:
        # 상태를 processing으로 변경
        pending.status = 'processing'
        pending.started_at = timezone.now()
        pending.save()
        
        return Response({
            'id': pending.id,
            'series_id': pending.series_id,
            'series_ids': pending.series_ids,
        })
    
    # 대기 중인 요청 없음
    return Response({'id': None})


@api_view(['POST'])
def complete_inference(request, request_id):
    """
    추론 완료 결과 저장
    연구실 PC 워커가 추론 완료 후 호출
    """
    try:
        inference = InferenceRequest.objects.get(id=request_id)
        
        # 결과 저장
        result_data = request.data
        
        if result_data.get('success'):
            inference.status = 'completed'
            inference.result = result_data
            inference.seg_instance_id = result_data.get('seg_instance_id')
        else:
            inference.status = 'failed'
            inference.result = result_data
        
        inference.completed_at = timezone.now()
        inference.save()
        
        return Response({'success': True})
        
    except InferenceRequest.DoesNotExist:
        return Response({'success': False, 'error': 'Request not found'}, status=404)


@api_view(['GET'])
def get_inference_status(request, series_id):
    """
    추론 상태 확인
    프론트엔드가 결과 대기 중 호출
    """
    try:
        inference = InferenceRequest.objects.get(series_id=series_id)
        
        return Response({
            'status': inference.status,
            'result': inference.result,
            'seg_instance_id': inference.seg_instance_id,
            'created_at': inference.created_at,
            'completed_at': inference.completed_at,
        })
        
    except InferenceRequest.DoesNotExist:
        return Response({'error': 'Not found'}, status=404)
```

#### 3. URL 설정

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    # 연구실 PC 워커용
    path('api/inference/pending', views.get_pending_inference, name='get_pending_inference'),
    path('api/inference/<int:request_id>/complete', views.complete_inference, name='complete_inference'),
    
    # 프론트엔드용
    path('api/inference/<str:series_id>/status', views.get_inference_status, name='get_inference_status'),
]
```

#### 4. 기존 추론 API 수정

```python
# 기존 추론 API (프론트엔드에서 호출)
@api_view(['POST'])
def start_inference(request, series_id):
    """
    추론 시작 (프론트엔드에서 호출)
    """
    series_ids = request.data.get('series_ids')  # 4개 시리즈 ID
    
    # DB에 요청 생성
    inference_request = InferenceRequest.objects.create(
        series_id=series_id,
        series_ids=series_ids,
        status='pending'
    )
    
    # 즉시 응답 (비동기 처리)
    return Response({
        'success': True,
        'request_id': inference_request.id,
        'message': '추론 요청이 생성되었습니다. 연구실 PC에서 처리 중입니다.',
    })
```

#### 5. 마이그레이션 및 배포

```bash
# 마이그레이션
python manage.py makemigrations
python manage.py migrate

# Gunicorn 재시작
sudo systemctl restart gunicorn
```

---

### 🟢 제가 할 일 (연구실 PC 워커)

#### 1. 워커 스크립트 수정

```python
# local_inference_worker.py (HTTP API 버전)
import requests
import time
import logging
from pathlib import Path

# 설정
GCP_API_URL = "http://34.42.223.43"  # GCP Django 서버 URL
POLL_INTERVAL = 30  # 30초마다 확인

logger = logging.getLogger(__name__)

def poll_for_requests():
    """GCP에서 대기 중인 요청 확인"""
    try:
        response = requests.get(
            f"{GCP_API_URL}/api/inference/pending",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"요청 확인 실패: {e}")
        return {'id': None}


def send_result(request_id, result):
    """추론 결과 전송"""
    try:
        response = requests.post(
            f"{GCP_API_URL}/api/inference/{request_id}/complete",
            json=result,
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"✅ 결과 전송 완료: {request_id}")
        return True
    except Exception as e:
        logger.error(f"결과 전송 실패: {e}")
        return False


def main():
    logger.info("🚀 HTTP API 워커 시작")
    logger.info(f"📡 GCP 서버: {GCP_API_URL}")
    logger.info(f"⏱️  폴링 간격: {POLL_INTERVAL}초")
    
    while True:
        try:
            # 1. 대기 중인 요청 확인
            request_data = poll_for_requests()
            
            if request_data.get('id'):
                logger.info(f"📋 새 요청 발견: {request_data['id']}")
                
                # 2. 추론 실행
                from local_inference import run_inference_local
                result = run_inference_local(
                    series_ids=request_data['series_ids']
                )
                
                # 3. 결과 전송
                send_result(request_data['id'], result)
            
            # 4. 다음 폴링까지 대기
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("워커 종료")
            break
        except Exception as e:
            logger.error(f"오류 발생: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
```

#### 2. 설정 파일 수정

```bash
# .env 파일에 추가
GCP_API_URL=http://34.42.223.43
POLL_INTERVAL=30
```

#### 3. 워커 재시작

```powershell
# 기존 워커 종료 (Ctrl+C)
# 새 워커 시작
python local_inference_worker.py
```

---

## 📋 구현 순서

### 1단계: GCP Django 설정 (조원님)
1. ✅ 모델 생성 (`InferenceRequest`)
2. ✅ API 엔드포인트 3개 생성
3. ✅ URL 설정
4. ✅ 마이그레이션
5. ✅ Gunicorn 재시작

### 2단계: 연구실 PC 워커 수정 (제가)
1. ✅ 워커 스크립트 HTTP API 버전으로 수정
2. ✅ 설정 파일 업데이트
3. ✅ 워커 재시작

### 3단계: 테스트
1. ✅ 프론트엔드에서 "AI 분석" 버튼 클릭
2. ✅ 워커 로그 확인
3. ✅ 결과 확인

---

## 🔍 API 명세

### 1. GET /api/inference/pending

**요청**: 없음

**응답**:
```json
{
  "id": 123,
  "series_id": "abc-def-ghi",
  "series_ids": ["id1", "id2", "id3", "id4"]
}
```

또는 (대기 중인 요청 없음):
```json
{
  "id": null
}
```

### 2. POST /api/inference/{request_id}/complete

**요청**:
```json
{
  "success": true,
  "seg_instance_id": "orthanc-instance-id",
  "tumor_detected": true,
  "tumor_volume_voxels": 1234,
  "inference_time_seconds": 45.2
}
```

**응답**:
```json
{
  "success": true
}
```

### 3. GET /api/inference/{series_id}/status

**요청**: 없음

**응답**:
```json
{
  "status": "completed",
  "result": { ... },
  "seg_instance_id": "orthanc-instance-id",
  "created_at": "2026-01-20T17:00:00Z",
  "completed_at": "2026-01-20T17:01:30Z"
}
```

---

## ✅ 장점

1. **간단함**: HTTP만 사용
2. **안정적**: 연구실 PC → GCP 단방향 통신
3. **방화벽 문제 없음**: ngrok 불필요
4. **추가 인프라 불필요**: Redis, 공유 폴더 불필요
5. **모니터링 쉬움**: Django Admin에서 요청 상태 확인 가능

---

## 📝 조원님께 전달할 내용

1. **Django 모델 코드** (위 코드 복사)
2. **API 엔드포인트 코드** (위 코드 복사)
3. **URL 설정** (위 코드 복사)
4. **마이그레이션 및 재시작 명령어**

**이 문서를 조원님께 전달하시면 됩니다!** 📤

---

*작성일: 2026-01-20 17:38*  
*작성자: AI Assistant*
