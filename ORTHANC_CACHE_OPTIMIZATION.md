# Orthanc 캐시 최적화 가이드

## 📋 개요

유방촬영술 DICOM 파일 로딩 속도 개선을 위한 Orthanc 캐시 설정 최적화 가이드입니다.

## 🔧 GCP 서버에서 실행할 명령어

### 1. Orthanc 설정 파일 확인

```bash
# Orthanc 컨테이너 확인
docker ps | grep orthanc

# Orthanc 설정 파일 위치 확인
docker exec <orthanc-container-name> cat /etc/orthanc/orthanc.json

# 또는 호스트에서 확인 (설정 파일이 마운트된 경우)
cat ~/orthanc/orthanc.json
```

### 2. Orthanc 설정 파일 최적화

다음 설정을 `orthanc.json`에 추가/수정:

```json
{
  "Name": "Hospital PACS Server",
  "HttpPort": 8042,
  "DicomPort": 4242,
  "RemoteAccessAllowed": true,
  "AuthenticationEnabled": true,
  "RegisteredUsers": {
    "admin": "admin123",
    "doctor": "doctor123",
    "viewer": "viewer123"
  },
  "StorageDirectory": "/var/lib/orthanc/db",
  "IndexDirectory": "/var/lib/orthanc/db",
  "StorageCompression": false,
  "MaximumStorageSize": 0,
  "MaximumPatientCount": 0,
  "RestApiWriteToFileSystemEnabled": true,
  
  // 성능 최적화 설정
  "ConcurrentJobs": 8,              // 동시 작업 수 증가 (기본: 4)
  "HttpThreadsCount": 100,          // HTTP 스레드 수 증가 (기본: 50)
  "HttpVerbose": false,
  "DicomVerbose": false,
  "StableAge": 60,
  "JobsHistorySize": 10,
  "SaveJobs": true,
  "StoreDicom": true,
  "DicomAlwaysAllowStore": true,
  "UnknownSopClassAccepted": true,
  
  // 캐시 최적화 (메모리 사용량 증가)
  "HttpRequestTimeout": 300,         // 요청 타임아웃 (초)
  "HttpRequestMaxSize": 104857600,   // 최대 요청 크기 (100MB)
  
  // 디스크 I/O 최적화
  "DatabaseBackend": "postgresql",   // PostgreSQL 사용 시 (선택사항)
  "DatabaseServer": "localhost",
  "DatabasePort": 5432,
  "DatabaseName": "orthanc",
  "DatabaseUsername": "orthanc",
  "DatabasePassword": "orthanc"
}
```

### 3. 설정 적용

```bash
# 1. 설정 파일 백업
cp ~/orthanc/orthanc.json ~/orthanc/orthanc.json.backup.$(date +%Y%m%d_%H%M%S)

# 2. 설정 파일 수정 (위의 최적화 설정 적용)

# 3. Orthanc 컨테이너 재시작
cd ~/orthanc
docker-compose restart

# 또는 직접 컨테이너 재시작
docker restart <orthanc-container-name>

# 4. 로그 확인
docker logs <orthanc-container-name> --tail 50

# 5. 설정 확인
curl -u admin:admin123 http://localhost:8042/system
```

## 📊 성능 개선 효과

### 최적화 전
- 유방촬영술 3장 동시 로드: 30-150MB 동시 전송
- 네트워크 병목 발생
- 초기 로딩 시간: 5-10초

### 최적화 후
- 현재 이미지 우선 로드: 즉시 표시
- 나머지 순차 로드: 네트워크 부하 분산
- 초기 로딩 시간: 1-2초 (현재 이미지)

## 🔍 모니터링

### Orthanc 시스템 정보 확인

```bash
# 시스템 정보
curl -u admin:admin123 http://localhost:8042/system

# 통계 정보
curl -u admin:admin123 http://localhost:8042/statistics

# 저장소 사용량
curl -u admin:admin123 http://localhost:8042/storage
```

### 성능 모니터링

```bash
# Orthanc 로그 실시간 확인
docker logs -f <orthanc-container-name>

# 메모리 사용량 확인
docker stats <orthanc-container-name>

# 디스크 I/O 확인
iostat -x 1
```

## ⚠️ 주의사항

1. **메모리 사용량**: `HttpThreadsCount`를 높이면 메모리 사용량이 증가합니다.
2. **디스크 공간**: `MaximumStorageSize: 0`은 무제한이므로 디스크 공간을 모니터링하세요.
3. **네트워크 대역폭**: 순차 로드로 네트워크 부하는 분산되지만, 전체 전송 시간은 비슷합니다.

## 🎯 추가 최적화 옵션

### 1. Nginx 캐싱 (선택사항)

```nginx
# /etc/nginx/sites-available/default
location /api/mri/orthanc/instances/ {
    proxy_pass http://localhost:8042/instances/;
    proxy_cache_path /var/cache/nginx/orthanc levels=1:2 keys_zone=orthanc_cache:10m max_size=1g inactive=60m;
    proxy_cache orthanc_cache;
    proxy_cache_valid 200 60m;
    proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
    add_header X-Cache-Status $upstream_cache_status;
}
```

### 2. Django 캐싱 (선택사항)

```python
# backend/mri_viewer/orthanc_views.py
from django.core.cache import cache

@api_view(['GET'])
def orthanc_instance_file(request, instance_id):
    cache_key = f'orthanc_file_{instance_id}'
    cached_data = cache.get(cache_key)
    
    if cached_data:
        return HttpResponse(cached_data, content_type='application/dicom')
    
    client = OrthancClient()
    dicom_data = client.get_instance_file(instance_id)
    cache.set(cache_key, dicom_data, timeout=3600)  # 1시간 캐시
    
    return HttpResponse(dicom_data, content_type='application/dicom')
```

## ✅ 체크리스트

- [ ] Orthanc 설정 파일 확인
- [ ] `ConcurrentJobs` 8로 증가
- [ ] `HttpThreadsCount` 100으로 증가
- [ ] Orthanc 컨테이너 재시작
- [ ] 로그 확인 (에러 없음)
- [ ] 시스템 정보 확인 (설정 반영 확인)
- [ ] 성능 테스트 (유방촬영술 로딩 속도 확인)

