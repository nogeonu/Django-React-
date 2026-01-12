#!/usr/bin/env python3
"""
병리 이미지 분류 Mosec 서비스 (포트 5008)
CLAM (Attention MIL) + H-optimus-0 Feature Extractor
2-class 분류: Normal vs Tumor
"""

import os
import io
import json
import logging
import numpy as np
import openslide
from PIL import Image
from typing import List, Dict
import tempfile

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from mosec import Server, Worker, get_logger

# 로깅 설정
logger = get_logger()
logger.setLevel(logging.INFO)

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 경로
MODEL_PATH = "/home/shrjsdn908/pathology_model/best_clam_model.pth"

# 클래스 이름
CLASS_NAMES = {
    0: 'Normal',
    1: 'Tumor'
}

# 전처리 설정
PATCH_SIZE = 224
TARGET_MAG = 20.0


class AttentionMIL(nn.Module):
    """CLAM-style Attention MIL 모델"""
    def __init__(self, input_dim=1536, hidden_dim=512, n_classes=2):
        super(AttentionMIL, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """
        Args:
            x: (N, input_dim) - N개의 패치 feature
        Returns:
            logits: (1, n_classes)
            attention: (1, N) - 각 패치의 attention weight
        """
        h = self.feature_extractor(x)  # (N, hidden_dim)
        A = self.attention(h)  # (N, 1)
        A = torch.transpose(A, 1, 0)  # (1, N)
        A = torch.softmax(A, dim=1)  # (1, N)
        M = torch.mm(A, h)  # (1, hidden_dim)
        logits = self.classifier(M)  # (1, n_classes)
        return logits, A


class WSIPatchDataset(Dataset):
    """WSI 패치 추출 Dataset (TCGA 학습 방식과 동일)"""
    def __init__(self, svs_path, patch_size=224, target_mag=20.0):
        self.wsi = openslide.OpenSlide(svs_path)
        self.patch_size = patch_size
        
        # 배율 계산
        mag = float(self.wsi.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
        self.scale = mag / target_mag
        
        # Tissue Masking (Thumbnail 기반)
        logger.info(f"🔍 Tissue masking 중...")
        thumb_width = self.wsi.dimensions[0] // 100
        thumb_height = self.wsi.dimensions[1] // 100
        thumb = self.wsi.get_thumbnail((thumb_width, thumb_height))
        thumb_gray = np.array(thumb.convert('L'))
        self.mask = thumb_gray < 235  # 조직 영역만 선택
        
        logger.info(f"📊 Tissue mask 크기: {self.mask.shape}")
        logger.info(f"📊 Tissue 비율: {self.mask.sum() / self.mask.size * 100:.2f}%")
        
        # 조직 영역에서만 패치 좌표 생성
        self.coords = []
        step = int(patch_size * self.scale)
        for y in range(0, self.wsi.dimensions[1] - step, step):
            for x in range(0, self.wsi.dimensions[0] - step, step):
                my = int(y / self.wsi.dimensions[1] * self.mask.shape[0])
                mx = int(x / self.wsi.dimensions[0] * self.mask.shape[1])
                if self.mask[my, mx]:  # 조직 영역인 경우만 추가
                    self.coords.append((x, y))
        
        logger.info(f"✅ 총 {len(self.coords)}개 패치 좌표 생성")
        
        # Transform (TCGA 데이터셋 통계 사용)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707, 0.578, 0.703),  # TCGA 통계
                std=(0.212, 0.230, 0.182)
            )
        ])
    
    def __len__(self):
        return len(self.coords)
    
    def __getitem__(self, idx):
        x, y = self.coords[idx]
        step = int(self.patch_size * self.scale)
        
        # 패치 읽기 및 리사이즈
        patch = self.wsi.read_region((x, y), 0, (step, step)).convert('RGB')
        patch = patch.resize((self.patch_size, self.patch_size), Image.LANCZOS)
        
        return self.transform(patch)


class PathologyWorker(Worker):
    """병리 이미지 분류 워커"""
    
    def __init__(self):
        super().__init__()
        self.clam_model = None
        self.backbone = None
        logger.info(f"💻 Device: {DEVICE}")
    
    def deserialize(self, data: bytes) -> dict:
        """요청 데이터 역직렬화"""
        try:
            json_data = json.loads(data.decode('utf-8'))
            logger.info(f"📥 수신한 데이터 키: {list(json_data.keys())}")
            return json_data
        except Exception as e:
            logger.error(f"❌ 역직렬화 오류: {str(e)}")
            raise
    
    def serialize(self, data) -> bytes:
        """결과 직렬화"""
        logger.info(f"📦 serialize 입력 타입: {type(data)}")
        
        # Mosec은 list를 전달할 수 있음
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                # list[dict] 형태 -> 첫 번째 dict 추출
                data = data[0]
            else:
                data = {"error": f"Unexpected list content: {type(data[0]) if data else 'empty'}"}
        elif not isinstance(data, dict):
            logger.error(f"❌ serialize 예상치 못한 데이터 타입: {type(data)}")
            data = {"error": f"Invalid data type: {type(data)}"}
        
        json_str = json.dumps(data)
        logger.info(f"📦 JSON 길이: {len(json_str)} bytes")
        return json_str.encode('utf-8')
    
    def forward(self, data) -> list:
        """
        병리 이미지 분류 추론
        
        Args:
            data: dict 또는 List[dict]
                {
                    "svs_file_path": "/path/to/svs/file.svs"
                }
        
        Returns:
            list: [{"results": {...}}]
        """
        # 데이터 추출
        if isinstance(data, list):
            request_data = data[0]
        else:
            request_data = data
        
        logger.info(f"📊 forward 입력 타입: {type(data)}")
        
        # 모델 로드 (첫 요청 시)
        if self.clam_model is None:
            logger.info(f"📦 모델 로딩 중...")
            
            # CLAM 모델 로드
            self.clam_model = AttentionMIL(input_dim=1536).to(DEVICE)
            self.clam_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.clam_model.eval()
            
            # H-optimus-0 백본 로드
            logger.info(f"🧠 H-optimus-0 백본 로딩 중...")
            
            # HuggingFace 토큰 확인 (선택적)
            hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
            if hf_token:
                logger.info(f"🔑 HuggingFace 토큰 사용")
                try:
                    from huggingface_hub import login
                    login(token=hf_token)
                except Exception as e:
                    logger.warning(f"⚠️ HuggingFace 로그인 실패: {str(e)}")
                    logger.info(f"💡 캐시에서 모델을 찾으려고 시도합니다...")
            else:
                logger.info(f"💡 토큰 없이 캐시에서 모델을 찾으려고 시도합니다...")
            
            try:
                self.backbone = timm.create_model(
                    "hf-hub:bioptimus/H-optimus-0",
                    pretrained=True,
                    init_values=1e-5
                ).to(DEVICE).eval()
                logger.info(f"✅ H-optimus-0 로드 성공!")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ H-optimus-0 로드 실패: {error_msg}")
                
                # Gated repo 에러인 경우
                if "401" in error_msg or "gated" in error_msg.lower() or "restricted" in error_msg.lower():
                    logger.error(f"💡 HuggingFace 토큰이 필요합니다!")
                    logger.error(f"💡 해결 방법:")
                    logger.error(f"   1. HF_TOKEN 환경변수 설정: export HF_TOKEN='your_token'")
                    logger.error(f"   2. 또는 한 번 다운로드: python3 -c \"from huggingface_hub import login; login(token='token'); import timm; timm.create_model('hf-hub:bioptimus/H-optimus-0', pretrained=True)\"")
                raise
            
            logger.info(f"✅ 모델 로드 완료: {MODEL_PATH}")
        
        try:
            # SVS 파일 경로 받기
            svs_file_path = request_data.get("svs_file_path", "")
            
            logger.info(f"📥 받은 svs_file_path: '{svs_file_path}' (타입: {type(svs_file_path)})")
            logger.info(f"📥 request_data 전체: {list(request_data.keys())}")
            
            if not svs_file_path:
                logger.error(f"❌ svs_file_path가 비어있습니다!")
                raise ValueError("svs_file_path가 전달되지 않았습니다.")
            
            if not os.path.exists(svs_file_path):
                logger.error(f"❌ 파일이 존재하지 않습니다: {svs_file_path}")
                # 디렉토리 확인
                if os.path.dirname(svs_file_path):
                    logger.info(f"📁 디렉토리 존재 여부: {os.path.exists(os.path.dirname(svs_file_path))}")
                raise ValueError(f"SVS 파일을 찾을 수 없습니다: {svs_file_path}")
            
            logger.info(f"✅ SVS 파일 경로 확인: {svs_file_path}")
            
            # 파일 크기 확인
            file_size = os.path.getsize(svs_file_path)
            logger.info(f"📊 SVS 파일 크기: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            # OpenSlide로 직접 열기 (파일 복사 불필요)
            tmp_path = svs_file_path
            
            # 패치 추출 (Tissue Masking 포함)
            logger.info(f"🔍 패치 추출 중...")
            dataset = WSIPatchDataset(tmp_path, patch_size=PATCH_SIZE, target_mag=TARGET_MAG)
            
            logger.info(f"📊 총 조직 패치 개수: {len(dataset)}")
            
            if len(dataset) == 0:
                logger.error(f"❌ 조직 패치가 없습니다! Tissue masking 결과 유효한 영역이 없습니다.")
                raise ValueError("조직 패치가 없습니다. 이미지가 대부분 배경입니다.")
            
            # Feature 추출 (배치 처리)
            loader = DataLoader(dataset, batch_size=128, shuffle=False)
            
            all_features = []
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    batch = batch.to(DEVICE)
                    
                    # H-optimus-0의 Feature 추출
                    try:
                        # 먼저 forward_features() 시도 (조원 코드 방식)
                        if hasattr(self.backbone, 'forward_features'):
                            outputs = self.backbone.forward_features(batch)
                        else:
                            # forward_features()가 없으면 forward() 사용
                            outputs = self.backbone(batch)
                        
                        # 출력 형태 확인 (첫 배치만)
                        if i == 0:
                            logger.info(f"🔍 Backbone 출력 형태: {outputs.shape if hasattr(outputs, 'shape') else type(outputs)}")
                            logger.info(f"🔍 Backbone 메서드: {'forward_features' if hasattr(self.backbone, 'forward_features') else 'forward'}")
                        
                        # Feature 추출
                        if hasattr(outputs, 'shape'):
                            if len(outputs.shape) == 3:
                                # (batch, tokens, features) 형태 - CLS token 추출
                                feats = outputs[:, 0].cpu()  # CLS token
                            elif len(outputs.shape) == 2:
                                # (batch, features) 형태 - 이미 pooling됨
                                feats = outputs.cpu()
                            else:
                                logger.error(f"❌ 예상치 못한 출력 형태: {outputs.shape}")
                                raise ValueError(f"Unexpected output shape: {outputs.shape}")
                        else:
                            logger.error(f"❌ 출력이 Tensor가 아닙니다: {type(outputs)}")
                            raise ValueError(f"Output is not a tensor: {type(outputs)}")
                        
                        # Feature 차원 확인
                        if i == 0:
                            logger.info(f"🔍 추출된 Feature 형태: {feats.shape}")
                        
                        all_features.append(feats)
                        
                    except Exception as e:
                        logger.error(f"❌ Feature 추출 오류 (배치 {i}): {str(e)}", exc_info=True)
                        raise  # 에러를 다시 발생시켜서 전체 프로세스 중단
            
            if len(all_features) == 0:
                logger.error(f"❌ Feature 추출 실패!")
                raise ValueError("Feature 추출에 실패했습니다.")
            
            slide_features = torch.cat(all_features, dim=0).to(DEVICE)  # (N, 1536)
            logger.info(f"✅ Feature 추출 완료: {slide_features.shape}")
            
            # CLAM 추론
            logger.info(f"🔮 CLAM 추론 중...")
            with torch.no_grad():
                logits, attention = self.clam_model(slide_features)
                probabilities = torch.softmax(logits, dim=1)[0]
                confidence, predicted_class = torch.max(probabilities, 0)
            
            # 결과 생성
            class_id = predicted_class.item()
            class_name = CLASS_NAMES[class_id]
            confidence_value = confidence.item()
            
            result = {
                'success': True,
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence_value,
                'probabilities': {
                    CLASS_NAMES[i]: float(probabilities[i].item())
                    for i in range(2)
                },
                'num_patches': len(slide_features),
                'top_attention_patches': attention[0].topk(5).indices.tolist()
            }
            
            logger.info(f"✅ 분류 완료: {class_name} (신뢰도: {confidence_value:.4f})")
            
        except Exception as e:
            logger.error(f"❌ 추론 오류: {str(e)}", exc_info=True)
            result = {
                'success': False,
                'error': str(e)
            }
        
        return [{"results": result}]


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🚀 병리 이미지 분류 Mosec 서비스 시작 (포트 5008)")
    logger.info("="*70)
    logger.info(f"📦 모델 경로: {MODEL_PATH}")
    logger.info(f"🔧 디바이스: {DEVICE}")
    logger.info(f"📊 클래스: {list(CLASS_NAMES.values())}")
    logger.info("="*70)
    logger.info("⚠️  명령줄 인자로 설정: --port 5008 --timeout 300000 --max-body-size 524288000")
    logger.info("="*70)
    
    server = Server()
    server.append_worker(
        PathologyWorker, 
        num=1, 
        max_batch_size=1,  # WSI는 크므로 배치 크기 1
        max_wait_time=120  # 120초 대기
    )
    server.run()  # 명령줄 인자는 Mosec이 자동으로 파싱

