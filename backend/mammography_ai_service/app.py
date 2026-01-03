"""
유방촬영술 AI 디텍션 서비스 (YOLO11 기반)
Mosec 프레임워크를 사용한 고성능 추론 서비스
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import io
import base64

# Mosec imports
from mosec import Server, Worker
import msgpack

# YOLO imports
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics 패키지가 설치되지 않았습니다. 설치 중...")
    os.system("pip install ultralytics opencv-python-headless")
    from ultralytics import YOLO

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수
MOSEC_PORT = int(os.getenv('MOSEC_PORT', 5004))  # 유방촬영술 AI는 5004 포트
MODEL_PATH = os.getenv(
    'MAMMOGRAPHY_MODEL_PATH',
    '/home/shrjsdn908/models/yolo11_mammography/best.pt'
)


class MammographyDetectionWorker(Worker):
    """YOLO11 유방촬영술 디텍션 워커"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = 'cuda' if self._check_cuda() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def _check_cuda(self):
        """CUDA 사용 가능 여부 확인"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """요청 데이터 역직렬화 (JSON과 msgpack 모두 지원)"""
        try:
            # JSON 시도
            import json
            return json.loads(data.decode('utf-8'))
        except:
            # msgpack 시도
            return msgpack.unpackb(data, raw=False)
    
    def serialize(self, data: Dict[str, Any]) -> bytes:
        """응답 데이터 직렬화"""
        return msgpack.packb(data, use_bin_type=True)
    
    def forward(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """YOLO11 추론 실행"""
        response = {
            "success": False,
            "instance_id": req.get("instance_id", ""),
            "detections": [],
            "annotated_image": "",
            "error": ""
        }
        
        try:
            # 모델 로드 (첫 요청 시)
            if self.model is None:
                logger.info(f"Loading YOLO11 model from {MODEL_PATH}")
                self.model = YOLO(MODEL_PATH)
                self.model.to(self.device)
                logger.info("✅ YOLO11 mammography detection model loaded successfully")
            
            # Base64 이미지 디코딩
            image_data = req.get("image_data", "")
            if not image_data:
                response["error"] = "No image_data provided"
                return response
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # YOLO 추론
            results = self.model.predict(
                source=image,
                conf=req.get("confidence", 0.25),
                iou=req.get("iou_threshold", 0.45),
                device=self.device,
                verbose=False
            )
            
            # 결과 파싱
            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(box.conf[0].cpu().numpy()),
                        'class_id': int(box.cls[0].cpu().numpy()),
                        'class_name': result.names[int(box.cls[0].cpu().numpy())]
                    }
                    detections.append(detection)
                
                # Annotated 이미지 생성
                annotated_img = result.plot()  # numpy array (BGR)
                annotated_pil = Image.fromarray(annotated_img[..., ::-1])  # BGR to RGB
                
                # PIL Image를 base64로 인코딩
                buffered = io.BytesIO()
                annotated_pil.save(buffered, format="PNG")
                annotated_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                response["annotated_image"] = annotated_base64
            
            response["success"] = True
            response["detections"] = detections
            logger.info(f"Detected {len(detections)} objects in instance {req.get('instance_id', '')}")
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}", exc_info=True)
            response["success"] = False
            response["error"] = str(e)
        
        return response


def main():
    """Mosec 서버 시작"""
    logger.info("🚀 Starting Mammography AI Detection Service (YOLO11)")
    logger.info(f"📦 Model path: {MODEL_PATH}")
    logger.info(f"🌐 Port: {MOSEC_PORT}")
    
    # 모델 파일 존재 확인
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    # Mosec은 환경변수로 포트를 설정합니다
    os.environ['MOSEC_PORT'] = str(MOSEC_PORT)
    
    # Mosec 서버 생성
    server = Server()
    server.append_worker(
        MammographyDetectionWorker,
        num=1,  # 워커 프로세스 수
        max_batch_size=1  # 배치 크기 (YOLO는 보통 1개씩 처리)
    )
    
    # 서버 시작 (포트는 환경변수에서 자동으로 읽음)
    server.run()


if __name__ == "__main__":
    main()
