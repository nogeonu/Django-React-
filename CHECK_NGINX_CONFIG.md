# Nginx 설정 확인 및 수정 가이드

## 문제
로그에 `client intended to send too large body: 10486321 bytes` 에러 발생
→ `client_max_body_size` 설정이 적용되지 않음

## 해결 방법

GCP 서버에서 다음 명령어 실행:

```bash
# 1. 현재 설정 파일 전체 확인
sudo cat /etc/nginx/sites-available/app

# 2. client_max_body_size가 있는지 확인
grep -n "client_max_body_size" /etc/nginx/sites-available/app

# 3. server 블록 내부에 제대로 있는지 확인
sudo nginx -T 2>&1 | grep -A 10 "server_name 34.42.223.43" | grep -A 5 "client_max_body_size"

# 4. 만약 없다면, server 블록 안에 추가
sudo nano /etc/nginx/sites-available/app
```

`server` 블록 내부에 다음 줄이 **반드시** 있어야 합니다:
```nginx
server {
    listen 80;
    server_name 34.42.223.43;
    
    client_max_body_size 500M;  # <-- 이 줄이 server 블록 안에 있어야 함
    
    root /srv/django-react/app/frontend/dist;
    # ... 나머지 설정
}
```

## 또는 한 번에 수정

```bash
# 기존 설정 확인
sudo cat /etc/nginx/sites-available/app | head -20

# client_max_body_size가 server 블록 안에 있는지 확인
sudo nginx -T 2>&1 | grep -B 5 -A 15 "server_name.*34.42.223.43"

# 만약 없다면 추가 (server_name 바로 다음에)
sudo sed -i '/server_name.*34.42.223.43;/a\    client_max_body_size 500M;' /etc/nginx/sites-available/app

# 설정 테스트
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx

# 확인
sudo nginx -T 2>&1 | grep -A 5 "server_name.*34.42.223.43"
```

## 전체 설정 파일 다시 작성 (확실한 방법)

```bash
# 백업
sudo cp /etc/nginx/sites-available/app /etc/nginx/sites-available/app.backup

# 전체 설정 파일 생성
sudo tee /etc/nginx/sites-available/app > /dev/null << 'EOF'
server {
    listen 80;
    server_name 34.42.223.43;

    client_max_body_size 500M;  # 중요: 이 줄 추가

    root /srv/django-react/app/frontend/dist;
    index index.html;
    
    charset utf-8;
    source_charset utf-8;

    # Orthanc PACS 프록시
    location /orthanc/ {
        proxy_pass http://localhost:8042/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
        proxy_request_buffering off;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /srv/django-react/app/backend/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location /media/ {
        alias /srv/django-react/app/backend/media/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        charset utf-8;
        default_type application/octet-stream;
    }

    location / {
        try_files $uri /index.html;
    }
}
EOF

# 설정 테스트
sudo nginx -t

# 재시작
sudo systemctl restart nginx

# 확인
sudo nginx -T 2>&1 | grep -A 3 "client_max_body_size"
```

