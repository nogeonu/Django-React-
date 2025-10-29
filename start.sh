#!/bin/bash

# EventEye 병원 관리 시스템 시작 스크립트

echo "🏥 EventEye 병원 관리 시스템을 시작합니다..."

# 백엔드 시작
echo "📡 Django 백엔드를 시작합니다..."
cd backend
python manage.py runserver &
BACKEND_PID=$!

# 프론트엔드 시작
echo "🎨 React 프론트엔드를 시작합니다..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo "✅ 서버가 시작되었습니다!"
echo "🌐 프론트엔드: http://localhost:3000"
echo "🔧 백엔드 API: http://localhost:8000"
echo "📊 Django 관리자: http://localhost:8000/admin"

# 프로세스 종료 함수
cleanup() {
    echo "🛑 서버를 종료합니다..."
    kill $BACKEND_PID $FRONTEND_PID
    exit
}

# Ctrl+C로 종료할 수 있도록 설정
trap cleanup SIGINT SIGTERM

# 프로세스가 실행 중인 동안 대기
wait
