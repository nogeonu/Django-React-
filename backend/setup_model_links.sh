#!/bin/bash

echo "=== 모델 심볼릭 링크 설정 ==="
echo ""

# 경로 설정
MODEL_DIR="/srv/django-react/app/backend/breast_ai_service/ml_model"
PERSISTENT_DIR="/opt/ml_models/breast_ai"

# ml_model 디렉토리 생성
mkdir -p "$MODEL_DIR"
echo "✅ 모델 디렉토리 생성: $MODEL_DIR"

# 영구 저장소 확인 및 생성
if [ ! -d "$PERSISTENT_DIR" ]; then
    echo "영구 저장소 생성: $PERSISTENT_DIR"
    sudo mkdir -p "$PERSISTENT_DIR"
    sudo chown -R $USER:$USER "$PERSISTENT_DIR"
fi

echo ""
echo "1. 세그멘테이션 모델 심볼릭 링크 설정:"
SEG_SOURCE="$PERSISTENT_DIR/unet_pytorch_best.pth"
SEG_LINK="$MODEL_DIR/unet_pytorch_best.pth"

# 원본 파일이 영구 저장소에 있는지 확인
if [ ! -f "$SEG_SOURCE" ]; then
    echo "   ⚠️  원본 파일이 없습니다: $SEG_SOURCE"
    echo "   모델 파일을 다음 경로에 복사해주세요:"
    echo "   sudo cp /path/to/unet_pytorch_best.pth $SEG_SOURCE"
else
    echo "   ✅ 원본 파일 확인: $SEG_SOURCE"
    # 기존 링크 제거
    if [ -L "$SEG_LINK" ] || [ -f "$SEG_LINK" ]; then
        rm -f "$SEG_LINK"
        echo "   기존 링크/파일 제거"
    fi
    # 심볼릭 링크 생성
    ln -s "$SEG_SOURCE" "$SEG_LINK"
    echo "   ✅ 심볼릭 링크 생성: $SEG_LINK -> $SEG_SOURCE"
fi

echo ""
echo "2. 분류 모델 심볼릭 링크 설정:"
CLS_SOURCE="$PERSISTENT_DIR/best_breast_mri_model.pth"
CLS_LINK="$MODEL_DIR/best_breast_mri_model.pth"

# 원본 파일이 영구 저장소에 있는지 확인
if [ ! -f "$CLS_SOURCE" ]; then
    echo "   ⚠️  원본 파일이 없습니다: $CLS_SOURCE"
    echo "   모델 파일을 다음 경로에 복사해주세요:"
    echo "   sudo cp /path/to/best_breast_mri_model.pth $CLS_SOURCE"
else
    echo "   ✅ 원본 파일 확인: $CLS_SOURCE"
    # 기존 링크 제거
    if [ -L "$CLS_LINK" ] || [ -f "$CLS_LINK" ]; then
        rm -f "$CLS_LINK"
        echo "   기존 링크/파일 제거"
    fi
    # 심볼릭 링크 생성
    ln -s "$CLS_SOURCE" "$CLS_LINK"
    echo "   ✅ 심볼릭 링크 생성: $CLS_LINK -> $CLS_SOURCE"
fi

echo ""
echo "3. 링크 확인:"
ls -lh "$MODEL_DIR"

echo ""
echo "4. 서비스 재시작:"
sudo systemctl restart breast-ai-service
echo "   ✅ 서비스 재시작 완료"

echo ""
echo "5. 서비스 상태 확인 (10초 대기 후):"
sleep 10
sudo systemctl status breast-ai-service --no-pager -l | head -30




