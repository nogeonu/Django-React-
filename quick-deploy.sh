#!/bin/bash

# 🚀 빠른 배포 스크립트 (GitHub Actions 우회)

set -e

echo "🚀 빠른 배포를 시작합니다..."
echo ""

# Git push
echo "📤 Git push..."
git push origin main
echo "✅ Git push 완료"
echo ""

# GCP 배포
echo "🔄 GCP 서버 배포 중..."

# SSH 설정 (필요시 수정)
GCP_HOST="34.42.223.43"
GCP_USER="${GCP_USER:-$(whoami)}"  # 기본값: 현재 사용자명

# gcloud가 있으면 사용, 없으면 직접 SSH 사용
if command -v gcloud &> /dev/null; then
    echo "📡 gcloud를 사용하여 배포합니다..."
    gcloud compute ssh koyang-2510 --zone=us-central1-a --command='
        set -e
        
        # 프로젝트 디렉토리 찾기
        if [ -d "/srv/django-react/app" ]; then
            PROJECT_DIR="/srv/django-react/app"
        elif [ -d "~/Django-React-" ]; then
            PROJECT_DIR="~/Django-React-"
        elif [ -d "~/Django-React--main" ]; then
            PROJECT_DIR="~/Django-React--main"
        else
            echo "❌ 프로젝트 디렉토리를 찾을 수 없습니다."
            exit 1
        fi
        
        cd $PROJECT_DIR
        echo "📂 작업 디렉토리: $PROJECT_DIR"
        
        echo "📥 코드 업데이트..."
        git pull origin main
        
        echo "🔧 백엔드 설정..."
        cd backend
        source .venv/bin/activate
        pip install -q --upgrade pip
        pip install -q -r requirements.txt
        python manage.py migrate --noinput
        python manage.py collectstatic --noinput
        
        echo "🎨 프론트엔드 빌드..."
        cd ../frontend
        
        # Node.js 메모리 제한 증가 (2GB)
        export NODE_OPTIONS="--max-old-space-size=2048"
        
        # npm ci 대신 npm install 사용 (메모리 효율적)
        npm install
        
        # 빌드 시도 (실패 시 기존 빌드 유지)
        if npm run build; then
            echo "✅ 프론트엔드 빌드 성공"
        else
            echo "⚠️  프론트엔드 빌드 실패 - 기존 빌드 유지"
            echo "💡 메모리 부족 시 서버 재시작 후 다시 시도하세요"
        fi
        
        echo "🔄 서비스 재시작..."
        sudo systemctl restart gunicorn
        sudo systemctl restart nginx
        sudo systemctl restart breast-ai-service || true
        
        echo "✅ 배포 완료!"
    '
else
    echo "📡 SSH를 사용하여 배포합니다..."
    echo "💡 GCP_USER 환경변수로 사용자명을 지정할 수 있습니다 (기본값: $GCP_USER)"
    
    ssh ${GCP_USER}@${GCP_HOST} << 'ENDSSH'
        set -e
        
        # 프로젝트 디렉토리 찾기
        if [ -d "/srv/django-react/app" ]; then
            PROJECT_DIR="/srv/django-react/app"
        elif [ -d "~/Django-React-" ]; then
            PROJECT_DIR="~/Django-React-"
        elif [ -d "~/Django-React--main" ]; then
            PROJECT_DIR="~/Django-React--main"
        else
            echo "❌ 프로젝트 디렉토리를 찾을 수 없습니다."
            exit 1
        fi
        
        cd $PROJECT_DIR
        echo "📂 작업 디렉토리: $PROJECT_DIR"
        
        echo "📥 코드 업데이트..."
        git pull origin main
        
        echo "🔧 백엔드 설정..."
        cd backend
        source .venv/bin/activate
        pip install -q --upgrade pip
        pip install -q -r requirements.txt
        python manage.py migrate --noinput
        python manage.py collectstatic --noinput
        
        echo "🎨 프론트엔드 빌드..."
        cd ../frontend
        
        # Node.js 메모리 제한 증가 (2GB)
        export NODE_OPTIONS="--max-old-space-size=2048"
        
        # npm ci 대신 npm install 사용 (메모리 효율적)
        npm install
        
        # 빌드 시도 (실패 시 기존 빌드 유지)
        if npm run build; then
            echo "✅ 프론트엔드 빌드 성공"
        else
            echo "⚠️  프론트엔드 빌드 실패 - 기존 빌드 유지"
            echo "💡 메모리 부족 시 서버 재시작 후 다시 시도하세요"
        fi
        
        echo "🔄 서비스 재시작..."
        sudo systemctl restart gunicorn
        sudo systemctl restart nginx
        sudo systemctl restart breast-ai-service || true
        
        echo "✅ 배포 완료!"
ENDSSH
fi

echo ""
echo "✅ 모든 배포가 완료되었습니다!"
echo "🌐 http://34.42.223.43"


