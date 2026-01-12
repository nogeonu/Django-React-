#!/bin/bash
# H-optimus-0 모델을 서버에 다운로드하는 스크립트
# 사용법: export HF_TOKEN="your_token" && bash setup_h_optimus.sh

# 환경변수에서 토큰 가져오기
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN}}"

if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN 환경변수가 설정되지 않았습니다."
    echo "💡 사용법: export HF_TOKEN='your_token' && bash setup_h_optimus.sh"
    exit 1
fi

echo "📥 H-optimus-0 모델 다운로드 중..."

# Python 스크립트 실행
python3 << EOF
import os
import torch
import timm

# HuggingFace 토큰 설정
hf_token = "${HF_TOKEN}"

# HuggingFace 로그인
from huggingface_hub import login
print("🔑 HuggingFace 로그인 중...")
try:
    login(token=hf_token)
    print("✅ 로그인 성공!")
except Exception as e:
    print(f"❌ 로그인 실패: {e}")
    exit(1)

# 모델 다운로드
print("📦 H-optimus-0 모델 다운로드 중...")
try:
    model = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0",
        pretrained=True,
        init_values=1e-5
    )
    print("✅ 모델 다운로드 완료!")
    print(f"📁 캐시 위치: ~/.cache/huggingface/hub/")
    print("💡 이제 토큰 없이도 모델을 사용할 수 있습니다.")
except Exception as e:
    print(f"❌ 모델 다운로드 실패: {e}")
    exit(1)
EOF

echo ""
echo "✅ 완료! 이제 pathology-mosec 서비스를 재시작하세요:"
echo "   sudo systemctl restart pathology-mosec"

