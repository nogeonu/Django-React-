#!/bin/bash
# 로컬 개발 환경에서 mosec 서비스 실행 스크립트

cd "$(dirname "$0")"
echo "📦 mosec 서비스 시작 중..."

# 가상환경 활성화 (있는 경우)
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

# mosec 및 의존성 설치 확인
if ! $PYTHON_CMD -c "import mosec" 2>/dev/null; then
    echo "⚠️  mosec이 설치되지 않았습니다. 설치 중..."
    $PIP_CMD install mosec torch torchvision msgspec
elif ! $PYTHON_CMD -c "import msgspec" 2>/dev/null; then
    echo "⚠️  msgspec이 설치되지 않았습니다. 설치 중..."
    $PIP_CMD install msgspec
fi

# 환경 변수 설정
export MOSEC_PORT=5003
export DL_MODEL_PATH="$(pwd)/ml_model/best_breast_mri_model.pth"

echo "🚀 mosec 서비스 시작: 포트 5003"
$PYTHON_CMD app.py

