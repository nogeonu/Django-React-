#!/bin/bash
# GCP 서버에서 실행할 빠른 수정 스크립트

set -e  # 오류 발생 시 중단

echo "=========================================="
echo "🚀 AI 서비스 빠른 수정 스크립트"
echo "=========================================="
echo ""

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 프로젝트 디렉토리 찾기
echo "1️⃣  프로젝트 디렉토리 찾기..."
PROJECT_DIR=$(find ~ -maxdepth 2 -name "Django-React-*" -type d 2>/dev/null | head -1)

if [ -z "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ 프로젝트 디렉토리를 찾을 수 없습니다.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 프로젝트 경로: $PROJECT_DIR${NC}"
echo ""

# 2. 기존 프로세스 종료
echo "2️⃣  기존 AI 서비스 프로세스 종료..."
pkill -f "mammography.*app.py" 2>/dev/null || true
pkill -f "mosec" 2>/dev/null || true
sleep 2
echo -e "${GREEN}✅ 완료${NC}"
echo ""

# 3. 가상환경 확인 및 활성화
echo "3️⃣  Python 가상환경 확인..."
cd "$PROJECT_DIR/backend"

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  가상환경이 없습니다. 생성 중...${NC}"
    python3 -m venv .venv
fi

source .venv/bin/activate
echo -e "${GREEN}✅ 가상환경 활성화 완료${NC}"
echo ""

# 4. 필수 패키지 설치
echo "4️⃣  필수 패키지 설치 중..."
pip install --upgrade pip -q
pip install ultralytics opencv-python torch torchvision mosec msgpack pydicom pillow -q
echo -e "${GREEN}✅ 패키지 설치 완료${NC}"
echo ""

# 5. 모델 디렉토리 및 파일 확인
echo "5️⃣  YOLO 모델 확인..."
MODEL_DIR="$HOME/models/yolo11_mammography"
MODEL_FILE="$MODEL_DIR/best.pt"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${YELLOW}⚠️  모델 파일이 없습니다. 기본 YOLO 모델 다운로드 중...${NC}"
    python3 << EOF
from ultralytics import YOLO
import os

model_dir = '$MODEL_DIR'
model_file = '$MODEL_FILE'

print(f"📥 Downloading YOLO model...")
model = YOLO('yolov8n.pt')
model.save(model_file)
print(f"✅ Model saved to {model_file}")
EOF
    echo -e "${GREEN}✅ 모델 다운로드 완료${NC}"
else
    echo -e "${GREEN}✅ 모델 파일 존재: $MODEL_FILE${NC}"
fi
echo ""

# 6. AI 서비스 디렉토리 확인
echo "6️⃣  AI 서비스 디렉토리 확인..."
AI_SERVICE_DIR="$PROJECT_DIR/backend/mammography_ai_service"

if [ ! -d "$AI_SERVICE_DIR" ]; then
    echo -e "${RED}❌ mammography_ai_service 디렉토리가 없습니다.${NC}"
    exit 1
fi

if [ ! -f "$AI_SERVICE_DIR/app.py" ]; then
    echo -e "${RED}❌ app.py 파일이 없습니다.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ AI 서비스 파일 확인 완료${NC}"
echo ""

# 7. 환경변수 설정 및 AI 서비스 실행
echo "7️⃣  AI 서비스 실행 중..."
cd "$AI_SERVICE_DIR"

# 환경변수 설정
export MAMMOGRAPHY_MODEL_PATH="$MODEL_FILE"
export MOSEC_PORT=5004

# 백그라운드로 실행
nohup python app.py > /tmp/mammography-ai.log 2>&1 &
AI_PID=$!

echo -e "${GREEN}✅ AI 서비스 시작 (PID: $AI_PID)${NC}"
echo ""

# 8. 서비스 시작 대기 및 확인
echo "8️⃣  서비스 시작 대기 중..."
sleep 5

# 프로세스 확인
if ps -p $AI_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✅ AI 서비스 프로세스 실행 중${NC}"
else
    echo -e "${RED}❌ AI 서비스 시작 실패${NC}"
    echo "로그 확인:"
    tail -20 /tmp/mammography-ai.log
    exit 1
fi

# 포트 확인
echo "9️⃣  포트 5004 확인 중..."
sleep 3

if netstat -tlnp 2>/dev/null | grep -q 5004; then
    echo -e "${GREEN}✅ 포트 5004가 정상적으로 열렸습니다.${NC}"
else
    echo -e "${YELLOW}⚠️  포트 5004가 아직 열리지 않았습니다. 조금 더 기다려주세요...${NC}"
    sleep 5
    if netstat -tlnp 2>/dev/null | grep -q 5004; then
        echo -e "${GREEN}✅ 포트 5004가 열렸습니다.${NC}"
    else
        echo -e "${RED}❌ 포트가 열리지 않았습니다. 로그를 확인하세요.${NC}"
        tail -30 /tmp/mammography-ai.log
        exit 1
    fi
fi
echo ""

# 10. 헬스 체크
echo "🔟 AI 서비스 헬스 체크..."
sleep 2

HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5004/health 2>/dev/null || echo "000")

if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo -e "${GREEN}✅ AI 서비스가 정상적으로 응답합니다!${NC}"
elif [ "$HEALTH_RESPONSE" = "000" ]; then
    echo -e "${YELLOW}⚠️  AI 서비스에 연결할 수 없습니다. (Mosec은 /health 엔드포인트가 없을 수 있음)${NC}"
    echo "   /inference 엔드포인트로 테스트해보세요."
else
    echo -e "${YELLOW}⚠️  AI 서비스 응답 코드: $HEALTH_RESPONSE${NC}"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}🎉 AI 서비스 설정 완료!${NC}"
echo "=========================================="
echo ""
echo "📊 서비스 정보:"
echo "   - PID: $AI_PID"
echo "   - 포트: 5004"
echo "   - 모델: $MODEL_FILE"
echo "   - 로그: /tmp/mammography-ai.log"
echo ""
echo "📝 유용한 명령어:"
echo "   - 로그 확인: tail -f /tmp/mammography-ai.log"
echo "   - 프로세스 확인: ps aux | grep app.py"
echo "   - 포트 확인: netstat -tlnp | grep 5004"
echo "   - 서비스 종료: pkill -f 'mammography.*app.py'"
echo ""
echo "🌐 브라우저에서 테스트:"
echo "   http://34.42.223.43/dicom-viewer/<instance-id>"
echo ""
