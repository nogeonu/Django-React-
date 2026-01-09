#!/bin/bash

echo "=== 배포 업데이트 스크립트 ==="
echo ""

# 1. 최신 코드 가져오기
echo "1. 최신 코드 가져오기..."
cd /srv/django-react/app
# git pull이 안되면 수동으로 확인
if [ -d ".git" ]; then
    git pull origin main || echo "⚠️  git pull 실패 (수동으로 확인 필요)"
else
    echo "⚠️  git 저장소가 아닙니다. 수동으로 코드를 확인하세요."
fi

# 2. 백엔드 디렉토리로 이동
cd /srv/django-react/app/backend

# 3. 가상환경 활성화
echo ""
echo "2. 가상환경 활성화..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✅ 가상환경 활성화 완료"
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ 가상환경 활성화 완료"
else
    echo "⚠️  가상환경을 찾을 수 없습니다."
    exit 1
fi

# 4. 새로운 라이브러리 설치
echo ""
echo "3. 새로운 라이브러리 설치..."
pip install plotly==5.18.0 scikit-image==0.21.0 scipy==1.11.4 --quiet
echo "✅ 라이브러리 설치 완료"

# 5. Django 마이그레이션 확인
echo ""
echo "4. Django 마이그레이션 확인..."
python manage.py migrate --noinput
echo "✅ 마이그레이션 완료"

# 6. 정적 파일 수집
echo ""
echo "5. 정적 파일 수집..."
python manage.py collectstatic --noinput
echo "✅ 정적 파일 수집 완료"

# 7. Django 서비스 재시작
echo ""
echo "6. Django 서비스 재시작..."
sudo systemctl restart gunicorn
sleep 3
sudo systemctl status gunicorn --no-pager -l | head -15

# 8. 딥러닝 서비스 상태 확인
echo ""
echo "7. 딥러닝 서비스 상태 확인..."
sudo systemctl status breast-ai-service --no-pager -l | head -15

# 9. 서비스 로그 확인 (모델 로드 확인)
echo ""
echo "8. 딥러닝 서비스 모델 로드 확인..."
sleep 5
sudo journalctl -u breast-ai-service --since "1 minute ago" --no-pager | grep -E "(세그멘테이션|분류|모델|로드|✅|❌|⚠️|🔄)" | tail -10

echo ""
echo "=== 배포 업데이트 완료 ==="









