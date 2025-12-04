#!/bin/bash

echo "=== 모델 파일 상태 확인 ==="
echo ""

# 현재 디렉토리 확인
CURRENT_DIR="/srv/django-react/app/backend/breast_ai_service"
MODEL_DIR="$CURRENT_DIR/ml_model"

echo "1. 모델 디렉토리 확인: $MODEL_DIR"
if [ -d "$MODEL_DIR" ]; then
    echo "   ✅ 디렉토리 존재"
    ls -lh "$MODEL_DIR" 2>/dev/null || echo "   ⚠️  디렉토리 내용 확인 실패"
else
    echo "   ❌ 디렉토리 없음"
fi

echo ""
echo "2. 세그멘테이션 모델 파일 확인:"
SEG_MODEL="$MODEL_DIR/unet_pytorch_best.pth"
if [ -f "$SEG_MODEL" ]; then
    echo "   ✅ 파일 존재: $SEG_MODEL"
    ls -lh "$SEG_MODEL"
    if [ -L "$SEG_MODEL" ]; then
        echo "   📎 심볼릭 링크입니다. 실제 경로:"
        readlink -f "$SEG_MODEL"
    fi
else
    echo "   ❌ 파일 없음: $SEG_MODEL"
    echo "   심볼릭 링크 확인:"
    if [ -L "$SEG_MODEL" ]; then
        echo "   📎 심볼릭 링크는 존재하지만 깨짐"
        readlink "$SEG_MODEL"
    fi
fi

echo ""
echo "3. 분류 모델 파일 확인:"
CLS_MODEL="$MODEL_DIR/best_breast_mri_model.pth"
if [ -f "$CLS_MODEL" ]; then
    echo "   ✅ 파일 존재: $CLS_MODEL"
    ls -lh "$CLS_MODEL"
    if [ -L "$CLS_MODEL" ]; then
        echo "   📎 심볼릭 링크입니다. 실제 경로:"
        readlink -f "$CLS_MODEL"
    fi
else
    echo "   ❌ 파일 없음: $CLS_MODEL"
    if [ -L "$CLS_MODEL" ]; then
        echo "   📎 심볼릭 링크는 존재하지만 깨짐"
        readlink "$CLS_MODEL"
    fi
fi

echo ""
echo "4. 영구 저장소 확인: /opt/ml_models/breast_ai/"
PERSISTENT_DIR="/opt/ml_models/breast_ai"
if [ -d "$PERSISTENT_DIR" ]; then
    echo "   ✅ 영구 저장소 존재"
    ls -lh "$PERSISTENT_DIR" 2>/dev/null || echo "   ⚠️  내용 확인 실패"
else
    echo "   ❌ 영구 저장소 없음"
fi

echo ""
echo "5. 서비스 상태 확인:"
systemctl status breast-ai-service --no-pager -l | head -20

echo ""
echo "6. 최근 서비스 로그 (모델 로드 관련):"
journalctl -u breast-ai-service --since "5 minutes ago" --no-pager | grep -E "(세그멘테이션|분류|모델|로드|✅|❌|⚠️)" | tail -20

