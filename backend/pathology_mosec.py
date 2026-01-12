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
from torch.utils.data import Dataset

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
    """WSI 패치 추출 Dataset"""
    def __init__(self, svs_path, patch_size=224, target_mag=20.0, max_patches=1000):
        self.wsi = openslide.OpenSlide(svs_path)
        self.patch_size = patch_size
        self.target_mag = target_mag
        self.max_patches = max_patches
        
        # 패치 좌표 생성
        self.patch_coords = self._generate_patch_coords()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _generate_patch_coords(self):
        """패치 좌표 생성 (간단한 그리드 샘플링)"""
        width, height = self.wsi.dimensions
        stride = self.patch_size * 2  # 50% 오버랩
        
        coords = []
        for y in range(0, height - self.patch_size, stride):
            for x in range(0, width - self.patch_size, stride):
                coords.append((x, y))
                if len(coords) >= self.max_patches:
                    return coords
        return coords
    
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        x, y = self.patch_coords[idx]
        patch = self.wsi.read_region((x, y), 0, (self.patch_size, self.patch_size))
        patch = patch.convert('RGB')
        
        # 배경 필터링 (흰색 배경 제거)
        patch_np = np.array(patch)
        if patch_np.mean() > 220:  # 대부분 흰색이면 스킵
            return None
        
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
    
    def serialize(self, data: dict) -> bytes:
        """결과 직렬화"""
        logger.info(f"📦 serialize 입력 타입: {type(data)}")
        
        if not isinstance(data, dict):
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
            self.backbone = timm.create_model(
                "hf-hub:bioptimus/H-optimus-0",
                pretrained=True,
                init_values=1e-5
            ).to(DEVICE).eval()
            
            logger.info(f"✅ 모델 로드 완료: {MODEL_PATH}")
        
        try:
            # SVS 파일 경로 받기
            svs_file_path = request_data.get("svs_file_path", "")
            
            if not svs_file_path or not os.path.exists(svs_file_path):
                raise ValueError(f"SVS 파일을 찾을 수 없습니다: {svs_file_path}")
            
            logger.info(f"📥 SVS 파일 경로: {svs_file_path}")
            
            # 파일 크기 확인
            file_size = os.path.getsize(svs_file_path)
            logger.info(f"📊 SVS 파일 크기: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            # OpenSlide로 직접 열기 (파일 복사 불필요)
            tmp_path = svs_file_path
            
            # 패치 추출
            logger.info(f"🔍 패치 추출 중...")
            dataset = WSIPatchDataset(tmp_path, patch_size=PATCH_SIZE, max_patches=1000)
            
            # Feature 추출
            logger.info(f"🧬 Feature 추출 중 ({len(dataset)} 패치)...")
            features = []
            with torch.no_grad():
                for i in range(len(dataset)):
                    patch = dataset[i]
                    if patch is None:
                        continue
                    patch = patch.unsqueeze(0).to(DEVICE)
                    feat = self.backbone(patch)  # (1, 1536)
                    features.append(feat.cpu())
            
            if len(features) == 0:
                raise ValueError("유효한 패치가 없습니다")
            
            features = torch.cat(features, dim=0)  # (N, 1536)
            logger.info(f"✅ Feature 추출 완료: {features.shape}")
            
            # CLAM 추론
            logger.info(f"🔮 CLAM 추론 중...")
            with torch.no_grad():
                features = features.to(DEVICE)
                logits, attention = self.clam_model(features)
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
                'num_patches': len(features),
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

