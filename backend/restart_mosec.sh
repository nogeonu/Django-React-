#!/bin/bash
# Mosec 서버 재시작 스크립트

echo "🔄 Mosec 서버 재시작 중..."

# 기존 프로세스 종료
echo "⏹️  기존 프로세스 종료 중..."
pkill -f "segmentation_mosec.py"
sleep 2

# 백업 생성
echo "💾 기존 파일 백업 중..."
cp /home/shrjsdn908/segmentation_mosec.py /home/shrjsdn908/segmentation_mosec_backup_$(date +%Y%m%d_%H%M%S).py

# 새 파일로 교체
echo "📝 새 파일로 교체 중..."
cp /home/shrjsdn908/segmentation_mosec_new.py /home/shrjsdn908/segmentation_mosec.py

# 새 프로세스 시작 (max-body-size 설정: 500MB)
echo "🚀 새 프로세스 시작 중 (max-body-size: 500MB)..."
nohup python3 /home/shrjsdn908/segmentation_mosec.py \
    --port 5006 \
    --max-body-size 524288000 \
    > /home/shrjsdn908/mosec.log 2>&1 &

sleep 3

# 프로세스 확인
if ps aux | grep -v grep | grep "segmentation_mosec.py" > /dev/null; then
    echo "✅ Mosec 서버 재시작 완료!"
    echo "📊 프로세스 정보:"
    ps aux | grep -v grep | grep "segmentation_mosec.py"
    echo ""
    echo "📋 로그 확인: tail -f /home/shrjsdn908/mosec.log"
else
    echo "❌ Mosec 서버 시작 실패!"
    echo "📋 로그 확인: cat /home/shrjsdn908/mosec.log"
    exit 1
fi
