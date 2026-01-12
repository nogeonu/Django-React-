#!/usr/bin/env python3
"""
H-optimus-0 모델 다운로드 스크립트 (메모리 최적화)
"""
import os
import sys
import gc

hf_token = os.environ.get('HF_TOKEN')

if not hf_token:
    print("❌ 오류: HF_TOKEN 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

try:
    from huggingface_hub import login
    print('🔑 HuggingFace 로그인 중...')
    login(token=hf_token)
    print('✅ 로그인 성공!')
except Exception as e:
    print('⚠️ 로그인 실패 (이미 로그인되어 있을 수 있음): ' + str(e))

try:
    import torch
    import timm
    
    torch.set_grad_enabled(False)
    
    print('📦 H-optimus-0 모델 다운로드 중... (메모리 최적화)')
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(
        repo_id='bioptimus/H-optimus-0',
        token=hf_token,
        local_files_only=False
    )
    print('✅ 모델 다운로드 완료! (캐시 경로: ' + str(cache_dir) + ')')
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
except Exception as e:
    print('⚠️ 모델 다운로드 실패 (이미 캐시에 있을 수 있음): ' + str(e))
    print('💡 계속 진행합니다...')
