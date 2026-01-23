"""
병리 이미지 로컬 추론 스크립트
Orthanc에서 원본 SVS 파일 경로를 찾아 추론 실행

사용법:
    python local_inference.py --instance-id <instance_id>
    
    또는
    
    python local_inference.py --instance-id <instance_id> --device cuda
"""
import sys
import os
from pathlib import Path
import requests
import logging
import argparse
from typing import Dict, Any, Optional
import glob

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 경로 설정
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# 환경 변수 또는 기본값
ORTHANC_URL = os.getenv("ORTHANC_URL", "http://34.42.223.43:8042")
ORTHANC_USER = os.getenv("ORTHANC_USER", "admin")
ORTHANC_PASSWORD = os.getenv("ORTHANC_PASSWORD", "admin123")

# 모델 경로 (CLAM 모델)
MODEL_PATH = os.getenv("MODEL_PATH", str(SRC_DIR / "best_model.pth"))

# GPU 확인
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        logger.info(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("ℹ️ GPU 사용 불가, CPU 모드로 실행")
except ImportError:
    logger.warning("⚠️ PyTorch가 설치되지 않았습니다.")
    GPU_AVAILABLE = False


def get_svs_file_path(instance_id: str) -> Optional[str]:
    """
    Orthanc에서 원본 SVS 파일 경로 찾기
    
    Args:
        instance_id: Orthanc instance ID
    
    Returns:
        원본 SVS 파일 경로 또는 None
    """
    try:
        # Orthanc에서 DICOM 메타데이터 조회
        logger.info(f"📥 Orthanc에서 DICOM 메타데이터 조회 중...")
        metadata_response = requests.get(
            f"{ORTHANC_URL}/instances/{instance_id}/tags?simplify",
            auth=(ORTHANC_USER, ORTHANC_PASSWORD),
            timeout=30
        )
        
        if metadata_response.status_code != 200:
            logger.error(f"❌ Orthanc 메타데이터 조회 실패: {metadata_response.status_code}")
            return None
        
        metadata = metadata_response.json()
        
        # Private Tag에서 원본 SVS 경로 추출 (0011,1001)
        original_svs_path = metadata.get('0011,1001')
        
        # Private Tag가 없으면 파일 시스템에서 검색
        if not original_svs_path:
            logger.warning(f"⚠️ DICOM에 원본 경로가 없습니다. 파일 시스템 검색 중...")
            
            # 환자 ID와 Series Description에서 파일명 추출
            patient_id = metadata.get('PatientID', '')
            series_desc = metadata.get('SeriesDescription', '')
            
            # Series Description에서 원본 파일명 추출 (예: "Pathology WSI - xxx.svs")
            if ' - ' in series_desc:
                original_filename = series_desc.split(' - ', 1)[1]
                
                # 파일 시스템에서 검색
                SVS_STORAGE_DIR = os.getenv('SVS_STORAGE_DIR', '/home/shrjsdn908/pathology_images')
                if os.path.exists(SVS_STORAGE_DIR):
                    # 패턴: {patient_id}_*_{original_filename}
                    pattern = os.path.join(SVS_STORAGE_DIR, f"{patient_id}_*_{original_filename}")
                    matching_files = glob.glob(pattern)
                    
                    if matching_files:
                        original_svs_path = matching_files[0]  # 첫 번째 매칭 파일 사용
                        logger.info(f"✅ 파일 시스템에서 발견: {original_svs_path}")
        
        if not original_svs_path or not os.path.exists(original_svs_path):
            logger.error(f"❌ 원본 SVS 파일을 찾을 수 없습니다: {original_svs_path}")
            return None
        
        logger.info(f"✅ 원본 SVS 파일 경로: {original_svs_path}")
        return original_svs_path
        
    except Exception as e:
        logger.error(f"❌ SVS 파일 경로 찾기 실패: {str(e)}")
        return None


def run_inference_local(filename: str, device: str = "cuda") -> Dict[str, Any]:
    """
    병리 이미지 로컬 추론 실행 (교육원 워커용)
    
    Args:
        filename: wsi/ 폴더에서 찾을 파일명 (예: "tumor_083.tif" 또는 "2024/01/case1.tif")
        device: 'cuda' or 'cpu'
    
    Returns:
        추론 결과 딕셔너리
    """
    try:
        # 1. wsi/ 폴더에서 파일 찾기
        WSI_DIR = Path(os.getenv("WSI_DIR", "wsi"))  # 기본값: 현재 디렉토리의 wsi/ 폴더
        svs_file_path = WSI_DIR / filename
        
        if not svs_file_path.exists():
            logger.error(f"❌ 파일을 찾을 수 없습니다: {svs_file_path}")
            return {
                'success': False,
                'error': f'파일을 찾을 수 없습니다: {filename} (wsi/ 폴더 확인 필요)',
                'class_id': None,
                'class_name': None,
                'confidence': 0.0,
                'probabilities': {},
                'num_patches': 0,
                'top_attention_patches': []
            }
        
        logger.info(f"✅ 파일 발견: {svs_file_path}")
        
        # 2. CLAM 모델로 추론 실행
        logger.info(f"🚀 추론 시작: {svs_file_path}")
        
        # CLAM 모델 추론 (실제 구현 필요)
        # 여기서는 인터페이스만 정의하고, 실제 모델 코드는 별도로 통합 필요
        result = run_clam_inference(str(svs_file_path), device)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 추론 실행 실패: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'class_id': None,
            'class_name': None,
            'confidence': 0.0,
            'probabilities': {},
            'num_patches': 0,
            'top_attention_patches': []
        }


def run_clam_inference(svs_file_path: str, device: str = "cuda") -> Dict[str, Any]:
    """
    CLAM 모델로 추론 실행
    
    Args:
        svs_file_path: 원본 SVS 파일 경로
        device: 'cuda' or 'cpu'
    
    Returns:
        추론 결과 딕셔너리
    
    Note:
        실제 CLAM 모델 코드를 여기에 통합해야 합니다.
        현재는 인터페이스만 정의되어 있습니다.
    """
    try:
        # TODO: CLAM 모델 코드 통합 필요
        # 예시 구조:
        # 1. SVS 파일 로드
        # 2. 패치 추출 (224x224, 최대 1000개)
        # 3. Feature 추출 (H-optimus-0)
        # 4. CLAM 모델로 분류
        # 5. Attention 패치 추출
        
        logger.warning("⚠️ CLAM 모델 코드가 아직 통합되지 않았습니다.")
        logger.warning("⚠️ 실제 추론을 위해서는 CLAM 모델 코드를 통합해야 합니다.")
        
        # 임시 더미 결과 (실제 구현 전까지)
        return {
            'success': False,
            'error': 'CLAM 모델 코드가 아직 통합되지 않았습니다.',
            'class_id': None,
            'class_name': None,
            'confidence': 0.0,
            'probabilities': {},
            'num_patches': 0,
            'top_attention_patches': []
        }
        
        # 실제 구현 예시 (주석 처리):
        # from src.models.clam_model import CLAMInference
        # 
        # model = CLAMInference(model_path=MODEL_PATH, device=device)
        # result = model.predict(svs_file_path)
        # 
        # return {
        #     'success': True,
        #     'class_id': result['class_id'],
        #     'class_name': result['class_name'],
        #     'confidence': result['confidence'],
        #     'probabilities': result['probabilities'],
        #     'num_patches': result['num_patches'],
        #     'top_attention_patches': result['top_attention_patches']
        # }
        
    except Exception as e:
        logger.error(f"❌ CLAM 추론 실패: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'class_id': None,
            'class_name': None,
            'confidence': 0.0,
            'probabilities': {},
            'num_patches': 0,
            'top_attention_patches': []
        }


def main():
    """명령줄 인터페이스"""
    parser = argparse.ArgumentParser(description="병리 이미지 로컬 추론")
    parser.add_argument("--filename", required=True, help="wsi/ 폴더에서 찾을 파일명 (예: tumor_083.tif)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="디바이스")
    
    args = parser.parse_args()
    
    # 추론 실행
    result = run_inference_local(args.filename, args.device)
    
    # 결과 출력
    if result.get('success'):
        print("\n✅ 추론 완료!")
        print(f"클래스: {result.get('class_name')}")
        print(f"신뢰도: {result.get('confidence'):.4f}")
        print(f"패치 수: {result.get('num_patches')}")
    else:
        print(f"\n❌ 추론 실패: {result.get('error')}")


if __name__ == "__main__":
    main()
