#!/bin/bash

# Orthanc Nginx 프록시 설정 스크립트

echo "🔧 Orthanc Nginx 프록시 설정을 시작합니다..."

# Nginx 설정 파일 백업
echo "📋 기존 설정 백업 중..."
sudo cp /etc/nginx/sites-available/default /etc/nginx/sites-available/default.backup.$(date +%Y%m%d_%H%M%S)

# Orthanc 프록시 설정 추가
echo "⚙️  Orthanc 프록시 설정 추가 중..."

sudo tee /etc/nginx/sites-available/orthanc-proxy.conf > /dev/null << 'EOF'
# Orthanc PACS 프록시 설정
location /orthanc/ {
    proxy_pass http://localhost:8042/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # WebSocket 지원
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    
    # 타임아웃 설정
    proxy_connect_timeout 600;
    proxy_send_timeout 600;
    proxy_read_timeout 600;
    send_timeout 600;
    
    # 버퍼 설정
    proxy_buffering off;
    proxy_request_buffering off;
}
EOF

# 기존 default 설정에 include 추가
echo "📝 메인 설정 파일 업데이트 중..."

# server 블록 내부에 include 추가
sudo sed -i '/location \/ {/i \    # Orthanc 프록시 설정 포함\n    include /etc/nginx/sites-available/orthanc-proxy.conf;\n' /etc/nginx/sites-available/default

# Nginx 설정 테스트
echo "🧪 Nginx 설정 테스트 중..."
if sudo nginx -t; then
    echo "✅ Nginx 설정이 올바릅니다!"
    
    # Nginx 재시작
    echo "🔄 Nginx 재시작 중..."
    sudo systemctl restart nginx
    
    echo ""
    echo "✅ Orthanc 프록시 설정이 완료되었습니다!"
    echo ""
    echo "🌐 이제 다음 URL로 접속할 수 있습니다:"
    echo "   http://34.42.223.43/orthanc/ui/app/#/"
    echo ""
    echo "🔗 또는 직접 접속:"
    echo "   http://34.42.223.43:8042"
    echo ""
else
    echo "❌ Nginx 설정에 오류가 있습니다!"
    echo "백업 파일로 복원합니다..."
    sudo cp /etc/nginx/sites-available/default.backup.* /etc/nginx/sites-available/default
    exit 1
fi

