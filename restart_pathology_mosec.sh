#!/bin/bash
# 병리 Mosec 서비스 재시작 스크립트

echo "🔄 병리 Mosec 서비스 재시작 중..."

# 서비스 중지
sudo systemctl stop pathology-mosec

# 잠시 대기
sleep 2

# 서비스 시작
sudo systemctl start pathology-mosec

# 상태 확인
sleep 2
sudo systemctl status pathology-mosec

echo "✅ 완료!"

