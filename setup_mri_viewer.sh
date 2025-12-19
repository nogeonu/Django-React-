#!/bin/bash

echo "=========================================="
echo "MRI 뷰어 설치 스크립트"
echo "=========================================="
echo ""

# 현재 디렉토리 저장
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/backend"

# 백엔드 디렉토리로 이동
cd "$BACKEND_DIR"

echo "1. 가상환경 활성화 중..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ 가상환경 활성화 완료"
else
    echo "✗ 가상환경을 찾을 수 없습니다. venv/bin/activate 파일이 없습니다."
    exit 1
fi

echo ""
echo "2. 필요한 패키지 설치 중..."
pip install nibabel==5.2.0
pip install SimpleITK==2.3.1
echo "✓ 패키지 설치 완료"

echo ""
echo "3. 데이터베이스 마이그레이션 실행 중..."
python manage.py migrate mri_viewer
echo "✓ 마이그레이션 완료"

echo ""
echo "=========================================="
echo "설치가 완료되었습니다!"
echo "=========================================="
echo ""
echo "다음 명령어로 서버를 실행하세요:"
echo ""
echo "백엔드:"
echo "  cd $BACKEND_DIR"
echo "  source venv/bin/activate"
echo "  python manage.py runserver 0.0.0.0:5000"
echo ""
echo "프론트엔드 (새 터미널):"
echo "  cd $SCRIPT_DIR/frontend"
echo "  npm run dev"
echo ""
echo "접속 URL: http://localhost:5173/mri-viewer"
echo ""

