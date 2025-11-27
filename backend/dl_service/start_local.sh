#!/bin/bash
# 로컬 개발 환경에서 mosec 서비스 실행 스크립트

cd "$(dirname "$0")"
echo "📦 mosec 서비스 시작 중..."

# 가상환경 활성화 (있는 경우)
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

# mosec 설치 확인
if ! python3 -c "import mosec" 2>/dev/null; then
    echo "⚠️  mosec이 설치되지 않았습니다. 설치 중..."
    pip3 install mosec torch torchvision
fi

# 환경 변수 설정
export MOSEC_PORT=5003
export DL_MODEL_PATH="$(pwd)/ml_model/best_breast_mri_model.pth"

echo "🚀 mosec 서비스 시작: 포트 5003"
python3 app.py

