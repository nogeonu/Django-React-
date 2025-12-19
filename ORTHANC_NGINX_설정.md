# 🌐 Orthanc Nginx 프록시 설정 가이드

## 📋 개요

Orthanc Web UI를 `/orthanc/` 경로로 접속할 수 있도록 Nginx 프록시를 설정합니다.

## 🚀 빠른 설정 (자동)

### GCP 서버에서 실행:

```bash
# 1. 서버 접속
gcloud compute ssh koyang-2510 --zone=us-central1-a

# 2. 프로젝트 디렉토리로 이동
cd ~/Django-React-

# 3. 최신 코드 가져오기
git pull origin main

# 4. 스크립트 실행
chmod +x setup-orthanc-proxy.sh
./setup-orthanc-proxy.sh
```

## 🔧 수동 설정

### 1. Nginx 설정 파일 백업

```bash
sudo cp /etc/nginx/sites-available/default /etc/nginx/sites-available/default.backup
```

### 2. Orthanc 프록시 설정 파일 생성

```bash
sudo nano /etc/nginx/sites-available/orthanc-proxy.conf
```

다음 내용 입력:

```nginx
# Orthanc PACS 프록시 설정
location /orthanc/ {
    proxy_pass http://localhost:8042/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # WebSocket 지원
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    
    # 타임아웃 설정
    proxy_connect_timeout 600;
    proxy_send_timeout 600;
    proxy_read_timeout 600;
    send_timeout 600;
    
    # 버퍼 설정
    proxy_buffering off;
    proxy_request_buffering off;
}
```

### 3. 메인 설정 파일에 포함

```bash
sudo nano /etc/nginx/sites-available/default
```

`server` 블록 내부에 추가:

```nginx
server {
    listen 80;
    server_name 34.42.223.43;
    
    # Orthanc 프록시 설정 포함
    include /etc/nginx/sites-available/orthanc-proxy.conf;
    
    location / {
        # 기존 설정...
    }
}
```

### 4. 설정 테스트 및 재시작

```bash
# 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx

# 상태 확인
sudo systemctl status nginx
```

## 🌐 접속 URL

설정 완료 후:

### Nginx 프록시 경유:
```
http://34.42.223.43/orthanc/ui/app/#/
```

### 직접 접속:
```
http://34.42.223.43:8042
```

## 🔍 문제 해결

### 1. 404 오류가 계속 나는 경우

```bash
# Nginx 에러 로그 확인
sudo tail -f /var/log/nginx/error.log

# Orthanc 컨테이너 확인
docker ps | grep orthanc

# Orthanc 로그 확인
docker logs $(docker ps -q --filter "ancestor=orthancteam/orthanc")
```

### 2. Nginx 설정 오류

```bash
# 설정 테스트
sudo nginx -t

# 백업으로 복원
sudo cp /etc/nginx/sites-available/default.backup /etc/nginx/sites-available/default
sudo systemctl restart nginx
```

### 3. 프록시 연결 실패

```bash
# Orthanc 포트 확인
curl http://localhost:8042/system

# 방화벽 확인
sudo ufw status
```

## 📊 설정 확인

### 테스트 명령어:

```bash
# 1. Nginx 설정 확인
sudo nginx -t

# 2. Orthanc 응답 확인
curl http://localhost:8042/system

# 3. 프록시 테스트
curl http://localhost/orthanc/system

# 4. 외부 접속 테스트 (로컬에서)
curl http://34.42.223.43/orthanc/system
```

## 🎯 완료 체크리스트

- [ ] Nginx 설정 파일 백업
- [ ] Orthanc 프록시 설정 추가
- [ ] Nginx 설정 테스트 통과
- [ ] Nginx 재시작 성공
- [ ] http://34.42.223.43/orthanc/ 접속 확인
- [ ] Orthanc Web UI 정상 작동 확인

## 💡 추가 정보

### Orthanc 기본 인증

Orthanc에 인증이 설정되어 있다면:

```nginx
location /orthanc/ {
    proxy_pass http://localhost:8042/;
    
    # 기본 인증 헤더 전달
    proxy_set_header Authorization $http_authorization;
    proxy_pass_header Authorization;
    
    # ... 나머지 설정
}
```

### HTTPS 설정 (선택사항)

Let's Encrypt로 HTTPS 설정:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 🔗 관련 문서

- [Orthanc Book - Nginx Configuration](https://book.orthanc-server.com/faq/nginx.html)
- [Nginx Reverse Proxy Guide](https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/)

## ✅ 완료!

설정이 완료되면 `/orthanc/` 경로로 Orthanc Web UI에 접속할 수 있습니다!

