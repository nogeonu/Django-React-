#!/bin/bash
# H-optimus-0 모델을 서버에 다운로드하는 스크립트

echo "📥 H-optimus-0 모델 다운로드 중..."

# Python 스크립트 실행
python3 << 'EOF'
import os
import torch
import timm

# HuggingFace 토큰 확인
hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')

if not hf_token:
    print("❌ HF_TOKEN 환경변수가 설정되지 않았습니다.")
    print("💡 사용법: export HF_TOKEN='your_token' && bash download_h_optimus.sh")
    exit(1)

# HuggingFace 로그인
from huggingface_hub import login
print("🔑 HuggingFace 로그인 중...")
login(token=hf_token)

# 모델 다운로드
print("📦 H-optimus-0 모델 다운로드 중...")
model = timm.create_model(
    "hf-hub:bioptimus/H-optimus-0",
    pretrained=True,
    init_values=1e-5
)

print("✅ 모델 다운로드 완료!")
print(f"📁 캐시 위치: ~/.cache/huggingface/hub/")
print("💡 이제 토큰 없이도 모델을 사용할 수 있습니다.")
EOF

