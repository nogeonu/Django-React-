# Orthanc 업로드 설정 확인 및 수정 가이드

## 문제
Orthanc Web UI (`http://34.42.223.43:8042/ui/app/#/`)에서 파일 업로드가 안 되는 경우

## 해결 방법

GCP 서버에서 다음 명령어 실행:

```bash
# 1. Orthanc 설정 파일 확인
cat ~/orthanc/orthanc.json

# 2. Orthanc 설정 파일 편집
nano ~/orthanc/orthanc.json
```

## 확인해야 할 설정

### 1. 업로드 크기 제한 확인
```json
{
  "MaximumStorageSize": 0,  // 0 = 무제한
  "StorageCompression": false
}
```

### 2. 인증 설정 확인
```json
{
  "AuthenticationEnabled": true,
  "RegisteredUsers": {
    "admin": "admin123",
    "doctor": "doctor123",
    "viewer": "viewer123"
  }
}
```

### 3. HTTP 설정 확인
```json
{
  "HttpPort": 8042,
  "RemoteAccessAllowed": true,
  "RestApiWriteToFileSystemEnabled": false  // false면 파일 시스템 쓰기 불가
}
```

## 권장 설정

전체 `orthanc.json` 파일 예시:

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
  "ConcurrentJobs": 4,
  "HttpVerbose": false,
  "DicomVerbose": false,
  "StableAge": 60,
  "JobsHistorySize": 10,
  "SaveJobs": true,
  "StoreDicom": true,
  "DicomAlwaysAllowStore": true,
  "UnknownSopClassAccepted": true,
  "HttpThreadsCount": 50
}
```

## 설정 적용

```bash
# 1. 설정 파일 백업
cp ~/orthanc/orthanc.json ~/orthanc/orthanc.json.backup

# 2. 설정 파일 수정 (위의 권장 설정 사용)

# 3. Orthanc 컨테이너 재시작
cd ~/orthanc
docker-compose restart

# 4. 로그 확인
docker logs hospital-orthanc --tail 50

# 5. 업로드 테스트
curl -u admin:admin123 http://localhost:8042/system
```

## 중요한 설정 항목

- **RestApiWriteToFileSystemEnabled**: `true`로 설정해야 API를 통해 파일 업로드 가능
- **StoreDicom**: `true`로 설정 (기본값)
- **DicomAlwaysAllowStore**: `true`로 설정 (DICOM 파일 저장 허용)
- **MaximumStorageSize**: `0` = 무제한 (필요시 조정)

## 문제 해결

만약 여전히 업로드가 안 된다면:

1. **Orthanc 로그 확인**:
   ```bash
   docker logs hospital-orthanc --tail 100
   ```

2. **권한 확인**:
   ```bash
   ls -la /var/lib/orthanc/db
   sudo chown -R 1000:1000 /var/lib/orthanc/db  # 필요시
   ```

3. **웹 브라우저 콘솔 확인**:
   - F12 → Console 탭에서 JavaScript 에러 확인
   - Network 탭에서 업로드 요청 상태 확인

4. **직접 API 테스트**:
   ```bash
   # 작은 테스트 파일 업로드
   curl -X POST -u admin:admin123 \
     -H "Content-Type: application/dicom" \
     --data-binary @test.dcm \
     http://localhost:8042/instances
   ```

