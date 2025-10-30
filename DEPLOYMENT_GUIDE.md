# 🚀 GCP VM 자동 배포 가이드

이 문서는 **VSCode → GitHub → GCP VM** 자동 배포 설정 방법을 설명합니다.

## 📋 목차
1. [전체 구조](#전체-구조)
2. [사전 준비](#사전-준비)
3. [GCP VM 설정](#gcp-vm-설정)
4. [GitHub Actions 설정](#github-actions-설정)
5. [Django/React 설정](#djangoreact-설정)
6. [배포 테스트](#배포-테스트)
7. [문제 해결](#문제-해결)
8. [새 프로젝트 배포](#새-프로젝트-배포)

---

## 전체 구조

```
개발자 PC (VSCode)
    ↓ git push
GitHub 저장소
    ↓ GitHub Actions
GCP VM (34.42.223.43)
    ↓ Nginx
사용자 브라우저
```

**흐름**: 코드 수정 → 커밋 → 푸시 → 자동 배포 → 서버 반영

---

## 사전 준비

### 1. 필요한 계정 및 서비스
- ✅ GitHub 계정
- ✅ Google Cloud Platform 계정
- ✅ GCP VM 인스턴스 (Ubuntu)
- ✅ 외부 고정 IP (34.42.223.43)

### 2. 로컬 환경 설정
```bash
# 프로젝트 디렉토리로 이동
cd "/Users/yourname/Desktop/건양대_바이오메디컬 /Django/Django-React--main"

# Git 초기화 (이미 되어있으면 생략)
git init
git branch -M main
```

### 3. GitHub 저장소 생성 및 연결
```bash
# GitHub에서 새 저장소 생성 후 연결
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 또는 이미 연결되어 있으면
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

---

## GCP VM 설정

### 1. SSH 접속
```bash
# 기본 키로 접속
ssh -i ~/.ssh/google_compute_engine shrjsdn908@34.42.223.43
```

### 2. 패키지 설치
```bash
# 패키지 업데이트 및 필수 프로그램 설치
sudo apt update && sudo apt upgrade -y
sudo apt install -y nginx python3-pip python3-venv git ufw nodejs npm
```

### 3. 방화벽 설정
```bash
# SSH, HTTP, HTTPS 허용
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable

# 상태 확인
sudo ufw status
```

### 4. 디렉토리 생성
```bash
# 프로젝트 디렉토리 생성
sudo mkdir -p /srv/django-react/app
sudo chown -R $USER:$USER /srv/django-react/app
```

### 5. Nginx 설정
```bash
# Nginx 설정 파일 생성
sudo tee /etc/nginx/sites-available/app >/dev/null <<'EOF'
server {
    listen 80;
    server_name 34.42.223.43;

    root /srv/django-react/app/frontend/dist;
    index index.html;

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        try_files $uri /index.html;
    }
}
EOF

# 설정 활성화
sudo ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app
sudo rm -f /etc/nginx/sites-enabled/default

# Nginx 테스트 및 재시작
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Gunicorn 서비스 설정
```bash
# Gunicorn systemd 서비스 생성
sudo tee /etc/systemd/system/gunicorn.service >/dev/null <<'EOF'
[Unit]
Description=Gunicorn for Django
After=network.target

[Service]
User=shrjsdn908
Group=shrjsdn908
WorkingDirectory=/srv/django-react/app/backend
Environment=PATH=/srv/django-react/app/backend/.venv/bin
ExecStart=/srv/django-react/app/backend/.venv/bin/gunicorn eventeye_backend.wsgi:application --bind 127.0.0.1:8000 --workers 3

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화
sudo systemctl daemon-reload
sudo systemctl enable gunicorn
```

### 7. SSH 키 설정 (GitHub Actions용)
```bash
# 기존 키 확인
ls -la ~/.ssh/

# GitHub에 올릴 개인키 복사 (Mac에서 실행)
# pbcopy < ~/.ssh/gcp_key
```

---

## GitHub Actions 설정

### 1. GitHub Secrets 등록

GitHub 저장소 → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

필수 시크릿 5개:

| 이름 | 값 | 예시 |
|------|-----|------|
| `SSH_HOST` | VM 외부 IP | `34.42.223.43` |
| `SSH_USER` | SSH 사용자명 | `shrjsdn908` |
| `SSH_PORT` | SSH 포트 | `22` |
| `SSH_KEY` | SSH 개인키 전체 | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `APP_DIR` | 서버 앱 경로 | `/srv/django-react/app` |

**SSH_KEY 등록 방법**:
```bash
# Mac 터미널에서
cat ~/.ssh/gcp_key | pbcopy

# 내용을 BEGIN/END 줄 포함 모두 복사해서 GitHub에 붙여넣기
```

---

## Django/React 설정

### 1. Django Settings 수정

**`backend/eventeye_backend/settings.py`** 수정 사항:

```python
# ALLOWED_HOSTS에 외부 IP 추가
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0', '34.42.223.43']

# INSTALLED_APPS에 앱 추가
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'django_filters',
    'hospital',
    'lung_cancer',        # 추가
    'medical_images',     # 추가
    'patients',           # 추가
]

# CORS 허용 도메인 추가
CORS_ALLOWED_ORIGINS = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "http://34.42.223.43",  # 추가
]

# 권한 설정 (공개 API)
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',  # IsAuthenticated → AllowAny
    ],
}

# 데이터베이스 설정
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    },
    'hospital_db': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'hospital_db',
        'USER': 'acorn',
        'PASSWORD': 'acorn1234',
        'HOST': '127.0.0.1',
        'PORT': '3306',
        'OPTIONS': {
            'charset': 'utf8mb4',
        },
    }
}
```

### 2. Django URLs 수정

**`backend/eventeye_backend/urls.py`** 수정:

```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('hospital.urls')),
    path('api/lung_cancer/', include('lung_cancer.urls')),  # 추가
]
```

### 3. React API 설정

**`frontend/src/lib/api.ts`** 수정:

```typescript
// baseURL을 빈 문자열로 (상대 경로 사용)
const API_BASE_URL = '';  # 'http://localhost:8002' → ''

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});
```

### 4. PostCSS 설정

**`frontend/postcss.config.js`** 수정:

```javascript
// ESM → CommonJS로 변경
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};
```

---

## 배포 테스트

### 1. 코드 수정 및 커밋
```bash
# 프로젝트 디렉토리로 이동
cd "/Users/yourname/Desktop/건양대_바이오메디컬 /Django/Django-React--main"

# 변경사항 확인
git status

# 변경사항 추가
git add .

# 커밋
git commit -m "설명 메시지"

# GitHub에 푸시 (자동 배포 트리거)
git push origin main
```

### 2. 배포 확인
1. **GitHub**: Actions 탭에서 배포 진행 상황 확인
2. **서버**: http://34.42.223.43 접속하여 홈페이지 확인

### 3. 수동 재배포 (필요시)
GitHub → Actions → **Run workflow** 버튼 클릭

---

## 문제 해결

### 1. 권한 오류 (Permission denied)
```bash
# VM에서 실행
sudo chown -R $USER:$USER /srv/django-react/app
```

### 2. Gunicorn 실행 안 됨
```bash
# 로그 확인
sudo journalctl -u gunicorn -n 50 --no-pager

# 재시작
sudo systemctl restart gunicorn
```

### 3. Nginx 에러
```bash
# 설정 테스트
sudo nginx -t

# 재시작
sudo systemctl restart nginx
```

### 4. API 500 에러
```bash
# Django 로그 확인
sudo journalctl -u gunicorn -n 100 --no-pager | grep -A 10 "Error\|Exception"

# DB 연결 확인
mysql -u acorn -pacorn1234 -e "SHOW DATABASES;"
```

### 5. 빌드 에러
- **TypeScript 에러**: 사용하지 않는 import 제거
- **PostCSS 에러**: `postcss.config.js`를 CJS 형식으로 변경
- **Node 버전**: Node 18.x 이상 필요

---

## 새 프로젝트 배포

### 핵심 요약
**같은 VM, 같은 SSH 키 사용** - 프로젝트만 다름!

### 빠른 설정 (5분)

1. **GitHub 저장소 생성**
```bash
cd /path/to/new-project
git init
git remote add origin https://github.com/YOUR_USERNAME/NEW_REPO.git
```

2. **워크플로 파일 복사**
- `.github/workflows/deploy.yml` 파일을 새 프로젝트에 복사

3. **GitHub Secrets 등록** (새 저장소마다)
- `SSH_KEY`: **기존 키 그대로** 재사용
- `APP_DIR`: `/srv/new-project/app` (프로젝트별로 변경)

4. **VM 설정**
```bash
sudo mkdir -p /srv/new-project/app
sudo chown -R shrjsdn908:shrjsdn908 /srv/new-project/app
```

5. **푸시**
```bash
git add .github/
git commit -m "Add deployment"
git push origin main
```

### 핵심 포인트

| 항목 | 재사용 여부 | 설명 |
|------|-----------|------|
| SSH 키 | ✅ 재사용 | 같은 키 계속 사용 |
| GitHub Secrets | ⚠️ 새로 등록 | 새 저장소마다 등록 필요 |
| VM IP | ✅ 재사용 | 같은 VM 사용 |
| APP_DIR | ❌ 변경 | 프로젝트별로 다름 |
| 포트 | ❌ 변경 | 다른 포트 사용 (81, 82...) |

---

## 주요 명령어 모음

### 로컬 (Mac)
```bash
# 커밋 및 푸시
git add .
git commit -m "메시지"
git push origin main

# SSH 키 확인
ls -la ~/.ssh/
cat ~/.ssh/gcp_key  # 개인키 전체 내용
```

### 서버 (VM)
```bash
# 서비스 상태 확인
sudo systemctl status gunicorn
sudo systemctl status nginx

# 로그 확인
sudo journalctl -u gunicorn -n 100
sudo journalctl -u nginx -n 100

# 재시작
sudo systemctl restart gunicorn
sudo systemctl restart nginx

# 프로젝트 확인
cd /srv/django-react/app
ls -la
```

---

## 📝 체크리스트

배포 전 확인사항:

- [ ] GCP VM 인스턴스 실행 중
- [ ] Nginx 설치 및 설정 완료
- [ ] Gunicorn 서비스 설정 완료
- [ ] GitHub Secrets 5개 모두 등록
- [ ] Django `settings.py` 설정 완료
- [ ] React API 설정 완료
- [ ] SSH 키 등록 완료

배포 후 확인:

- [ ] GitHub Actions 성공
- [ ] http://34.42.223.43 접속 가능
- [ ] API 요청 정상 작동
- [ ] 데이터베이스 연결 정상

---

## 🎯 완료!

이제 **`git push origin main`** 한 번으로 자동 배포가 완료됩니다! 🚀

---

**작성일**: 2025-10-30
**환경**: GCP VM (Ubuntu), Python 3.10, Node.js 18.x, Django 4.2, React + Vite

