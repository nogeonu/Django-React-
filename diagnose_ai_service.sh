#!/bin/bash

echo "=========================================="
echo "AI 서비스 진단 스크립트"
echo "=========================================="
echo ""

# 1. AI 서비스 프로세스 확인
echo "1. AI 서비스 프로세스 확인..."
ps aux | grep -E "mammography|mosec|python.*app.py" | grep -v grep
echo ""

# 2. 포트 5003 사용 확인
echo "2. 포트 5003 사용 확인..."
sudo netstat -tlnp | grep 5003 || echo "포트 5003이 사용되지 않음"
echo ""

# 3. Systemd 서비스 상태 확인
echo "3. Systemd 서비스 상태 확인..."
sudo systemctl status mammography-ai.service 2>/dev/null || echo "mammography-ai.service 없음"
sudo systemctl status breast-ai-service.service 2>/dev/null || echo "breast-ai-service.service 없음"
echo ""

# 4. AI 서비스 로그 확인
echo "4. AI 서비스 로그 확인 (최근 20줄)..."
if [ -f /var/log/mammography-ai.log ]; then
    tail -20 /var/log/mammography-ai.log
elif [ -f /var/log/breast-ai.log ]; then
    tail -20 /var/log/breast-ai.log
else
    echo "로그 파일을 찾을 수 없음"
fi
echo ""

# 5. AI 모델 파일 확인
echo "5. AI 모델 파일 확인..."
find ~/Django-React-* -name "*.pt" -o -name "*.pth" 2>/dev/null | head -10
echo ""

# 6. Python 가상환경 확인
echo "6. Python 가상환경 확인..."
ls -la ~/Django-React-*/backend/.venv/bin/python* 2>/dev/null || echo "가상환경 없음"
echo ""

# 7. 필수 패키지 확인
echo "7. 필수 패키지 확인..."
if [ -d ~/Django-React-*/backend/.venv ]; then
    source ~/Django-React-*/backend/.venv/bin/activate 2>/dev/null
    pip list | grep -E "ultralytics|torch|mosec|flask" || echo "필수 패키지 미설치"
    deactivate 2>/dev/null
fi
echo ""

# 8. AI 서비스 디렉토리 확인
echo "8. AI 서비스 디렉토리 확인..."
ls -la ~/Django-React-*/backend/mammography_ai_service/ 2>/dev/null || echo "mammography_ai_service 디렉토리 없음"
ls -la ~/Django-React-*/backend/mri_viewer/ 2>/dev/null || echo "mri_viewer 디렉토리 없음"
echo ""

echo "=========================================="
echo "진단 완료"
echo "=========================================="
