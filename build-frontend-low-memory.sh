#!/bin/bash

# 메모리 부족 환경에서 프론트엔드 빌드를 위한 스크립트

echo "🔧 메모리 최적화 빌드 시작..."
echo ""

# 현재 메모리 상태 확인
echo "📊 현재 메모리 상태:"
free -h
echo ""

# Swap 파일 확인 및 생성 (없는 경우)
if [ ! -f /swapfile ]; then
    echo "💾 Swap 파일이 없습니다. 생성을 권장합니다."
    echo "다음 명령어로 2GB swap 생성 가능:"
    echo "  sudo fallocate -l 2G /swapfile"
    echo "  sudo chmod 600 /swapfile"
    echo "  sudo mkswap /swapfile"
    echo "  sudo swapon /swapfile"
    echo ""
fi

# Node.js 프로세스 정리
echo "🧹 기존 Node.js 프로세스 정리..."
pkill -f node || true
sleep 2

# 프론트엔드 디렉토리로 이동
cd ~/Django-React-/frontend

# Node.js 메모리 제한 설정
export NODE_OPTIONS="--max-old-space-size=2048"

echo "🎨 프론트엔드 빌드 시작..."
echo "   메모리 제한: 2GB"
echo ""

# 의존성 설치 (이미 설치되어 있으면 빠름)
echo "📦 의존성 확인..."
npm install --prefer-offline --no-audit

# 빌드 실행
echo "🏗️  빌드 실행..."
if npm run build; then
    echo ""
    echo "✅ 빌드 성공!"
    echo ""
    
    # 빌드 결과 확인
    if [ -d "dist" ]; then
        echo "📁 빌드 결과:"
        du -sh dist
        echo ""
    fi
else
    echo ""
    echo "❌ 빌드 실패"
    echo ""
    echo "💡 해결 방법:"
    echo "1. Swap 메모리 추가 (위 명령어 참고)"
    echo "2. 서버 재시작 후 다시 시도"
    echo "3. 불필요한 프로세스 종료"
    echo ""
    exit 1
fi

echo "✅ 완료!"
