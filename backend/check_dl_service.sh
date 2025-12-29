#!/bin/bash
# 딥러닝 서비스 상태 확인 스크립트
# GCP VM에서 실행: bash check_dl_service.sh

echo "=========================================="
echo "🔍 딥러닝 서비스 상태 확인"
echo "=========================================="

# 1. 서비스 상태 확인
echo ""
echo "1️⃣ Systemd 서비스 상태:"
if systemctl is-active --quiet breast-ai-service; then
    echo "   ✅ breast-ai-service: 실행 중"
    systemctl status breast-ai-service --no-pager -l | head -10
else
    echo "   ❌ breast-ai-service: 실행되지 않음"
    echo "   시작하려면: sudo systemctl start breast-ai-service"
fi

# 2. 포트 확인
echo ""
echo "2️⃣ 포트 5003 확인:"
if lsof -i :5003 > /dev/null 2>&1; then
    echo "   ✅ 포트 5003: 사용 중"
    lsof -i :5003 | head -5
else
    echo "   ❌ 포트 5003: 사용되지 않음"
fi

# 3. 프로세스 확인
echo ""
echo "3️⃣ Python 프로세스 확인:"
ps aux | grep -E "mosec|breast_ai_service|app.py" | grep -v grep || echo "   ❌ 관련 프로세스 없음"

# 4. 헬스 체크
echo ""
echo "4️⃣ 서비스 헬스 체크:"
HEALTH_URL="http://127.0.0.1:5003/health"
if curl -s --max-time 5 "$HEALTH_URL" > /dev/null; then
    echo "   ✅ 헬스 체크 성공"
    curl -s "$HEALTH_URL" | python3 -m json.tool 2>/dev/null || curl -s "$HEALTH_URL"
else
    echo "   ❌ 헬스 체크 실패 (서비스가 응답하지 않음)"
fi

# 5. 로그 확인 (최근 20줄)
echo ""
echo "5️⃣ 최근 서비스 로그:"
if [ -f /var/log/breast-ai-service.log ]; then
    echo "   로그 파일: /var/log/breast-ai-service.log"
    tail -20 /var/log/breast-ai-service.log
elif journalctl -u breast-ai-service -n 20 --no-pager > /dev/null 2>&1; then
    echo "   Systemd 저널 로그:"
    journalctl -u breast-ai-service -n 20 --no-pager | tail -20
else
    echo "   로그를 찾을 수 없습니다."
fi

# 6. 모델 파일 확인
echo ""
echo "6️⃣ 모델 파일 확인:"
MODEL_PATH="/srv/django-react/app/backend/breast_ai_service/ml_model/best_breast_mri_model.pth"
if [ -f "$MODEL_PATH" ]; then
    echo "   ✅ 모델 파일 존재: $MODEL_PATH"
    ls -lh "$MODEL_PATH"
else
    echo "   ❌ 모델 파일 없음: $MODEL_PATH"
fi

echo ""
echo "=========================================="
echo "✅ 확인 완료"
echo "=========================================="
echo ""
echo "💡 문제 해결 방법:"
echo "   1. 서비스 시작: sudo systemctl start breast-ai-service"
echo "   2. 서비스 재시작: sudo systemctl restart breast-ai-service"
echo "   3. 서비스 상태: sudo systemctl status breast-ai-service"
echo "   4. 로그 확인: sudo journalctl -u breast-ai-service -f"




