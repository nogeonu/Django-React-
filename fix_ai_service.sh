#!/bin/bash

echo "=========================================="
echo "AI 서비스 수정 및 재시작 스크립트"
echo "=========================================="
echo ""

# 프로젝트 경로 찾기
PROJECT_DIR=$(find ~ -maxdepth 2 -name "Django-React-*" -type d | head -1)

if [ -z "$PROJECT_DIR" ]; then
    echo "❌ 프로젝트 디렉토리를 찾을 수 없습니다."
    exit 1
fi

echo "✅ 프로젝트 경로: $PROJECT_DIR"
echo ""

# 1. 기존 AI 서비스 프로세스 종료
echo "1. 기존 AI 서비스 프로세스 종료..."
pkill -f "mammography.*app.py" 2>/dev/null
pkill -f "mosec" 2>/dev/null
sleep 2
echo "✅ 완료"
echo ""

# 2. 가상환경 활성화
echo "2. Python 가상환경 활성화..."
cd "$PROJECT_DIR/backend"
if [ ! -d ".venv" ]; then
    echo "⚠️  가상환경이 없습니다. 생성 중..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "✅ 완료"
echo ""

# 3. 필수 패키지 설치
echo "3. 필수 패키지 설치..."
pip install --upgrade pip -q
pip install ultralytics opencv-python flask flask-cors torch torchvision -q
echo "✅ 완료"
echo ""

# 4. AI 서비스 파일 확인 및 수정
echo "4. AI 서비스 파일 확인..."
AI_SERVICE_DIR="$PROJECT_DIR/backend/mammography_ai_service"

if [ ! -d "$AI_SERVICE_DIR" ]; then
    echo "❌ mammography_ai_service 디렉토리가 없습니다."
    exit 1
fi

if [ ! -f "$AI_SERVICE_DIR/app.py" ]; then
    echo "❌ app.py 파일이 없습니다."
    exit 1
fi

echo "✅ AI 서비스 파일 확인 완료"
echo ""

# 5. YOLO 모델 파일 확인
echo "5. YOLO 모델 파일 확인..."
MODEL_FILE=$(find "$PROJECT_DIR" -name "best.pt" -o -name "yolo*.pt" | head -1)

if [ -z "$MODEL_FILE" ]; then
    echo "⚠️  YOLO 모델 파일을 찾을 수 없습니다."
    echo "   기본 YOLO 모델을 다운로드합니다..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
else
    echo "✅ 모델 파일 발견: $MODEL_FILE"
fi
echo ""

# 6. AI 서비스 실행
echo "6. AI 서비스 실행..."
cd "$AI_SERVICE_DIR"

# 백그라운드로 실행
nohup python app.py > /tmp/mammography-ai.log 2>&1 &
AI_PID=$!

sleep 3

# 프로세스 확인
if ps -p $AI_PID > /dev/null; then
    echo "✅ AI 서비스 시작 성공 (PID: $AI_PID)"
    echo "   로그 파일: /tmp/mammography-ai.log"
else
    echo "❌ AI 서비스 시작 실패"
    echo "   로그 확인:"
    tail -20 /tmp/mammography-ai.log
    exit 1
fi
echo ""

# 7. 포트 확인
echo "7. 포트 5003 확인..."
sleep 2
if netstat -tlnp 2>/dev/null | grep -q 5003; then
    echo "✅ 포트 5003이 정상적으로 열렸습니다."
else
    echo "⚠️  포트 5003이 열리지 않았습니다. 로그를 확인하세요."
    tail -20 /tmp/mammography-ai.log
fi
echo ""

# 8. 테스트 요청
echo "8. AI 서비스 테스트..."
sleep 2
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5003/health 2>/dev/null)

if [ "$RESPONSE" = "200" ]; then
    echo "✅ AI 서비스가 정상적으로 응답합니다."
elif [ "$RESPONSE" = "000" ]; then
    echo "⚠️  AI 서비스에 연결할 수 없습니다."
else
    echo "⚠️  AI 서비스 응답 코드: $RESPONSE"
fi
echo ""

echo "=========================================="
echo "AI 서비스 설정 완료"
echo "=========================================="
echo ""
echo "📝 추가 확인 사항:"
echo "   - 로그 확인: tail -f /tmp/mammography-ai.log"
echo "   - 프로세스 확인: ps aux | grep app.py"
echo "   - 포트 확인: netstat -tlnp | grep 5003"
echo ""
