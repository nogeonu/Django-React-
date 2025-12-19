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
gcloud compute ssh koyang-2510 --zone=us-central1-a --command='
    set -e
    cd ~/Django-React-
    
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
    npm ci
    npm run build
    
    echo "🔄 서비스 재시작..."
    sudo systemctl restart gunicorn
    sudo systemctl restart nginx
    sudo systemctl restart breast-ai-service || true
    
    echo "✅ 배포 완료!"
'

echo ""
echo "✅ 모든 배포가 완료되었습니다!"
echo "🌐 http://34.42.223.43"

