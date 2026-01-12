#!/bin/bash
# 원격 서버에서 H-optimus-0 모델을 다운로드하는 개선된 스크립트

# HuggingFace 토큰은 환경변수로 설정하거나 스크립트 실행 시 전달해야 합니다
# 사용법: HF_TOKEN='your_token' ./setup_h_optimus_remote.sh
# 또는: export HF_TOKEN='your_token' && ./setup_h_optimus_remote.sh

if [ -z "$HF_TOKEN" ]; then
    echo "❌ 오류: HF_TOKEN 환경변수가 설정되지 않았습니다."
    echo "💡 사용법: HF_TOKEN='your_token' ./setup_h_optimus_remote.sh"
    exit 1
fi

ssh -i ~/.ssh/gcp_deploy_key -o StrictHostKeyChecking=no shrjsdn908@34.42.223.43 "export HF_TOKEN='$HF_TOKEN' && cd /srv/django-react/app/backend && source .venv/bin/activate && python3 << 'EOF'

import os
import sys
from pathlib import Path

print('=' * 60)
print('🔍 환경 확인 중...')
print('=' * 60)

# 필수 패키지 확인
required_packages = {
    'timm': 'timm',
    'huggingface_hub': 'huggingface_hub',
    'torch': 'torch',
    'tqdm': 'tqdm'
}

missing_packages = []
for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
        print(f'✅ {package_name} 설치됨')
    except ImportError:
        print(f'❌ {package_name} 설치되지 않음')
        missing_packages.append(package_name)

if missing_packages:
    packages_str = ', '.join(missing_packages)
    print(f'\n⚠️  다음 패키지들을 설치해주세요: {packages_str}')
    print('   pip install ' + ' '.join(missing_packages))
    sys.exit(1)

print('\n' + '=' * 60)
print('🔑 HuggingFace 로그인 중...')
print('=' * 60)

try:
    from huggingface_hub import login
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        print('❌ 오류: HF_TOKEN 환경변수가 설정되지 않았습니다.')
        sys.exit(1)
    login(token=hf_token)
    print('✅ 로그인 성공!')
except Exception as e:
    print(f'❌ 로그인 실패: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n' + '=' * 60)
print('📦 H-optimus-0 모델 다운로드 중...')
print('=' * 60)

try:
    import timm
    import torch
    from huggingface_hub import snapshot_download
    
    model_id = 'bioptimus/H-optimus-0'
    
    print(f'\n📥 모델 저장소: {model_id}')
    print('⏳ 다운로드 시작... (이 작업은 몇 분이 걸릴 수 있습니다)')
    print('💡 진행 상황이 자동으로 표시됩니다.\n')
    
    # 진행 상황을 보여주며 모델 다운로드
    try:
        # snapshot_download은 내부적으로 tqdm을 사용하므로 진행 상황이 자동으로 표시됩니다
        sys.stdout.flush()
        
        hf_token = os.environ.get('HF_TOKEN')
        cache_dir = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            local_files_only=False,
            resume_download=True
        )
        print(f'\n✅ 모델 다운로드 완료!')
        print(f'📁 다운로드 위치: {cache_dir}')
        
        # 다운로드된 파일 크기 확인
        if cache_dir and Path(cache_dir).exists():
            all_files = [f for f in Path(cache_dir).rglob('*') if f.is_file()]
            total_size = sum(f.stat().st_size for f in all_files)
            file_count = len(all_files)
            print(f'📊 다운로드된 파일 수: {file_count}')
            print(f'💾 총 크기: {total_size / (1024**3):.2f} GB')
    except Exception as download_error:
        print(f'⚠️  snapshot_download 실패, timm으로 직접 다운로드 시도: {download_error}')
        print('   (timm이 자동으로 다운로드합니다)')
        cache_dir = None
    
    print('\n🔄 모델 로드 중...')
    print('   (모델 가중치를 메모리에 로드하는 중...)')
    model = timm.create_model('hf-hub:bioptimus/H-optimus-0', pretrained=True, init_values=1e-5)
    print('✅ 모델 로드 완료!')
    
    # 캐시 위치 확인 (snapshot_download을 사용하지 않은 경우)
    if not cache_dir:
        cache_dir_path = Path.home() / '.cache' / 'huggingface' / 'hub'
        if cache_dir_path.exists():
            print(f'\n📁 캐시 위치: {cache_dir_path}')
            # 모델 파일 크기 확인
            model_files = list(cache_dir_path.rglob('*bioptimus*H-optimus-0*'))
            if model_files:
                total_size = sum(f.stat().st_size for f in model_files if f.is_file())
                print(f'📊 다운로드된 파일 수: {len(model_files)}')
                print(f'💾 총 크기: {total_size / (1024**3):.2f} GB')
            else:
                print('⚠️  모델 파일을 찾을 수 없습니다. (캐시 위치 확인 필요)')
    
    # 모델 정보 출력
    print(f'\n📋 모델 정보:')
    print(f'   - 타입: {type(model).__name__}')
    print(f'   - 디바이스: {next(model.parameters()).device}')
    
    # 모델 파라미터 수 확인
    try:
        param_count = sum(p.numel() for p in model.parameters())
        print(f'   - 파라미터 수: {param_count / 1e6:.2f}M')
    except:
        pass
    
    print('\n💡 이제 토큰 없이도 모델을 사용할 수 있습니다.')
    
except Exception as e:
    print(f'❌ 모델 다운로드 실패: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n' + '=' * 60)
print('✅ 모든 작업 완료!')
print('=' * 60)
print('\n💡 다음 단계:')
print('   sudo systemctl restart pathology-mosec')

EOF
"

