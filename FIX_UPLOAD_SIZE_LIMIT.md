# 업로드 크기 제한 수정 가이드

HTTP 413 (Request Entity Too Large) 에러를 해결하기 위한 설정입니다.

## 문제
- NIfTI 파일이나 큰 DICOM 파일 업로드 시 413 에러 발생
- Nginx와 Django 모두에서 업로드 크기 제한이 필요

## 해결 방법

### 1. Django 설정 (이미 적용됨)
`backend/eventeye/settings.py`에 다음 설정이 추가되었습니다:
```python
DATA_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024  # 500 MB
FILE_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024  # 500 MB
```

### 2. Nginx 설정 (GCP 서버에서 수정 필요)

GCP 서버에서 다음 명령어 실행:

```bash
# Nginx 설정 파일 확인
sudo cat /etc/nginx/sites-available/app | grep -i "client_max_body_size"

# Nginx 설정 파일 수정
sudo nano /etc/nginx/sites-available/app
```

`server` 블록 안에 다음 줄이 있는지 확인하고, 없으면 추가:
```nginx
client_max_body_size 500M;
```

예시:
```nginx
server {
    listen 80;
    server_name 34.42.223.43;
    
    client_max_body_size 500M;  # 이 줄 추가 또는 수정
    
    # ... 나머지 설정
}
```

저장 후:
```bash
# 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx
```

### 3. Gunicorn 설정 (필요시)

만약 gunicorn을 사용하는 경우, systemd 서비스 파일도 확인:

```bash
sudo nano /etc/systemd/system/gunicorn.service
```

LimitRequestBody 설정이 있다면 제거하거나 충분히 큰 값으로 설정:
```
# LimitRequestBody=0  # 무제한 (또는 충분히 큰 값)
```

그리고 재시작:
```bash
sudo systemctl daemon-reload
sudo systemctl restart gunicorn
```

## 테스트

업로드를 다시 시도하고, 413 에러가 발생하지 않는지 확인하세요.

## 참고
- 500MB는 예시 값입니다. 필요에 따라 조정 가능합니다.
- 더 큰 파일이 필요한 경우 값을 증가시킬 수 있습니다 (예: 1G, 2G).

