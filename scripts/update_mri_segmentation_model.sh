#!/bin/bash
# MRI 세그멘테이션 모델 업데이트 스크립트
# 기존 96×96×96 모델을 128×128×128 모델로 교체

set -e

# 설정
GCP_USER="shrjsdn908"
GCP_HOST="34.42.223.43"  # GCP 서버 IP
GCP_MODEL_DIR="/home/shrjsdn908/models/mri_models"
LOCAL_MODEL_PATH="/Users/nogeon-u/Desktop/건양대_바이오메디컬/Django/mri_segmantesion/best_model.pth"
OLD_MODEL_NAME="Phase1_Segmentation_best.pth"
NEW_MODEL_NAME="best_model.pth"

echo "=========================================="
echo "MRI 세그멘테이션 모델 업데이트"
echo "=========================================="
echo "기존 모델: ${OLD_MODEL_NAME} (96×96×96)"
echo "새 모델: ${NEW_MODEL_NAME} (128×128×128)"
echo ""

# 1. 로컬 모델 파일 확인
if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "❌ 오류: 로컬 모델 파일을 찾을 수 없습니다: $LOCAL_MODEL_PATH"
    exit 1
fi

echo "✅ 로컬 모델 파일 확인: $(ls -lh "$LOCAL_MODEL_PATH" | awk '{print $5}')"
echo ""

# 2. GCP 서버에 SSH로 접속하여 기존 모델 백업 및 삭제
echo "📡 GCP 서버에 연결 중..."
ssh ${GCP_USER}@${GCP_HOST} << 'ENDSSH'
    set -e
    MODEL_DIR="/home/shrjsdn908/models/mri_models"
    OLD_MODEL="Phase1_Segmentation_best.pth"
    NEW_MODEL="best_model.pth"
    
    echo "📁 모델 디렉토리 확인: $MODEL_DIR"
    
    # 디렉토리 생성 (없으면)
    mkdir -p "$MODEL_DIR"
    
    # 기존 모델 백업 (있으면)
    if [ -f "$MODEL_DIR/$OLD_MODEL" ]; then
        echo "📦 기존 모델 백업 중..."
        mv "$MODEL_DIR/$OLD_MODEL" "$MODEL_DIR/${OLD_MODEL}.backup.$(date +%Y%m%d_%H%M%S)"
        echo "✅ 기존 모델 백업 완료"
    else
        echo "ℹ️  기존 모델 파일이 없습니다 (이미 삭제되었거나 처음 설치)"
    fi
    
    # 새 모델이 이미 있으면 백업
    if [ -f "$MODEL_DIR/$NEW_MODEL" ]; then
        echo "📦 기존 새 모델 백업 중..."
        mv "$MODEL_DIR/$NEW_MODEL" "$MODEL_DIR/${NEW_MODEL}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    echo "✅ GCP 서버 준비 완료"
ENDSSH

if [ $? -ne 0 ]; then
    echo "❌ GCP 서버 연결 실패"
    exit 1
fi

# 3. 새 모델 업로드
echo ""
echo "📤 새 모델 업로드 중..."
scp "$LOCAL_MODEL_PATH" ${GCP_USER}@${GCP_HOST}:${GCP_MODEL_DIR}/${NEW_MODEL_NAME}

if [ $? -eq 0 ]; then
    echo "✅ 새 모델 업로드 완료"
else
    echo "❌ 모델 업로드 실패"
    exit 1
fi

# 4. GCP 서버에서 파일 확인
echo ""
echo "🔍 업로드된 모델 확인 중..."
ssh ${GCP_USER}@${GCP_HOST} << ENDSSH
    MODEL_DIR="/home/shrjsdn908/models/mri_models"
    NEW_MODEL="best_model.pth"
    
    if [ -f "\$MODEL_DIR/\$NEW_MODEL" ]; then
        echo "✅ 모델 파일 확인:"
        ls -lh "\$MODEL_DIR/\$NEW_MODEL"
        echo ""
        echo "📊 모델 정보:"
        file "\$MODEL_DIR/\$NEW_MODEL"
    else
        echo "❌ 모델 파일을 찾을 수 없습니다"
        exit 1
    fi
ENDSSH

echo ""
echo "=========================================="
echo "✅ 모델 업데이트 완료!"
echo "=========================================="
echo ""
echo "⚠️  다음 단계:"
echo "1. segmentation_mosec.py 서비스 재시작 필요"
echo "   sudo systemctl restart segmentation-service"
echo "2. 서비스 상태 확인"
echo "   sudo systemctl status segmentation-service"
echo ""
