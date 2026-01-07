#!/bin/bash

echo "=========================================="
echo "Systemd 서비스 생성 스크립트"
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

# 현재 사용자 확인
CURRENT_USER=$(whoami)
echo "✅ 현재 사용자: $CURRENT_USER"
echo ""

# Systemd 서비스 파일 생성
SERVICE_FILE="/tmp/mammography-ai.service"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Mammography AI Detection Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR/backend/mammography_ai_service
Environment="PATH=$PROJECT_DIR/backend/.venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$PROJECT_DIR/backend/.venv/bin/python app.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/mammography-ai.log
StandardError=append:/var/log/mammography-ai-error.log

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Systemd 서비스 파일 생성 완료"
echo ""

# 서비스 파일 이동 (sudo 필요)
echo "서비스 파일을 시스템에 설치합니다..."
echo "sudo 권한이 필요합니다."
echo ""

sudo cp "$SERVICE_FILE" /etc/systemd/system/mammography-ai.service
sudo chmod 644 /etc/systemd/system/mammography-ai.service

# 로그 파일 생성
sudo touch /var/log/mammography-ai.log
sudo touch /var/log/mammography-ai-error.log
sudo chown $CURRENT_USER:$CURRENT_USER /var/log/mammography-ai.log
sudo chown $CURRENT_USER:$CURRENT_USER /var/log/mammography-ai-error.log

# Systemd 리로드
sudo systemctl daemon-reload

echo "✅ Systemd 서비스 설치 완료"
echo ""

# 서비스 시작
echo "서비스를 시작합니다..."
sudo systemctl start mammography-ai.service
sleep 3

# 서비스 상태 확인
sudo systemctl status mammography-ai.service --no-pager

echo ""
echo "=========================================="
echo "Systemd 서비스 설정 완료"
echo "=========================================="
echo ""
echo "📝 유용한 명령어:"
echo "   - 서비스 시작: sudo systemctl start mammography-ai.service"
echo "   - 서비스 중지: sudo systemctl stop mammography-ai.service"
echo "   - 서비스 재시작: sudo systemctl restart mammography-ai.service"
echo "   - 서비스 상태: sudo systemctl status mammography-ai.service"
echo "   - 부팅 시 자동 시작: sudo systemctl enable mammography-ai.service"
echo "   - 로그 확인: sudo journalctl -u mammography-ai.service -f"
echo ""
