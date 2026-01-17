#!/bin/bash
# FastAPI 약물 검색 서비스 systemd 설정 스크립트

set -e

APP_DIR="${1:-/srv/django-react/app}"
USER="${2:-shrjsdn908}"

echo "🔧 FastAPI 약물 검색 서비스 설정 중..."

# FastAPI 서비스 디렉토리 확인
DRUG_API_DIR="$APP_DIR/backend/drug_api"
if [ ! -d "$DRUG_API_DIR" ]; then
    echo "❌ FastAPI 서비스 디렉토리가 없습니다: $DRUG_API_DIR"
    echo "   CDSS_Final_Package/backend 파일들을 $DRUG_API_DIR에 복사해주세요."
    exit 1
fi

# systemd 서비스 파일 생성
sudo tee /etc/systemd/system/drug-api-service.service > /dev/null <<EOF
[Unit]
Description=FastAPI Drug Search and Interaction Service
After=network.target mysql.service

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$DRUG_API_DIR
Environment="PATH=$APP_DIR/backend/.venv/bin"
Environment="PYTHONPATH=$APP_DIR/backend"
ExecStart=$APP_DIR/backend/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8002 --workers 2
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 서비스 활성화 및 시작
sudo systemctl daemon-reload
sudo systemctl enable drug-api-service
sudo systemctl restart drug-api-service || sudo systemctl start drug-api-service

echo "✅ FastAPI 약물 검색 서비스 설정 완료!"
echo "📋 서비스 상태 확인: sudo systemctl status drug-api-service"
echo "📋 로그 확인: sudo journalctl -u drug-api-service -f"
