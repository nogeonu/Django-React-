"""
유방촬영술 AI 디텍션 API Views
"""
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import base64
import os
import logging
from io import BytesIO
from PIL import Image
import numpy as np
import pydicom

from .orthanc_client import OrthancClient

logger = logging.getLogger(__name__)

# YOLO 모델 경로
MODEL_PATH = os.getenv(
    'MAMMOGRAPHY_MODEL_PATH',
    '/home/shrjsdn908/models/yolo11_mammography/best.pt'
)

# YOLO 모델 로드 (전역 변수로 한 번만 로드)
_yolo_model = None

def get_yolo_model():
    """YOLO 모델 로드 (싱글톤 패턴)"""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from {MODEL_PATH}")
            _yolo_model = YOLO(MODEL_PATH)
            logger.info("✅ YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    return _yolo_model


@api_view(['POST'])
def mammography_ai_detection(request, instance_id):
    """
    유방촬영술 이미지에 대해 YOLO11 디텍션 실행 (Django에서 직접 실행)
    
    POST /api/mri/mammography/instances/<instance_id>/detect/
    
    Request body:
    {
        "confidence": 0.25,  # optional
        "iou_threshold": 0.45  # optional
    }
    
    Response:
    {
        "success": true,
        "instance_id": "abc123",
        "detections": [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.85,
                "class_id": 0,
                "class_name": "mass"
            }
        ],
        "annotated_image_base64": "base64_string"
    }
    """
    try:
        logger.info(f"Starting YOLO detection for instance {instance_id}")
        
        # Orthanc에서 DICOM 이미지 가져오기
        client = OrthancClient()
        dicom_data = client.get_instance_file(instance_id)
        
        # DICOM을 PIL Image로 변환
        dicom = pydicom.dcmread(BytesIO(dicom_data))
        pixel_array = dicom.pixel_array
        
        # 정규화 (0-255)
        if pixel_array.max() > 255:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # PIL Image로 변환
        if len(pixel_array.shape) == 2:  # Grayscale
            image = Image.fromarray(pixel_array, mode='L').convert('RGB')
        else:
            image = Image.fromarray(pixel_array)
        
        # YOLO 모델 로드
        model = get_yolo_model()
        
        # YOLO 추론 실행
        confidence = float(request.data.get('confidence', 0.25))
        iou_threshold = float(request.data.get('iou_threshold', 0.45))
        
        logger.info(f"Running YOLO inference with conf={confidence}, iou={iou_threshold}")
        results = model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            device='cpu',  # GCP VM은 CPU 사용
            verbose=False
        )
        
        # 결과 파싱
        detections = []
        annotated_image_base64 = ""
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
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
            buffered = BytesIO()
            annotated_pil.save(buffered, format="PNG")
            annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        logger.info(f"✅ Detection completed: {len(detections)} objects found")
        
        # 응답 구성
        response_data = {
            'success': True,
            'instance_id': instance_id,
            'detections': detections,
            'annotated_image_base64': annotated_image_base64,
            'error': ''
        }
        
        return Response(response_data)
        
    except Exception as e:
        logger.error(f"❌ Detection failed: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'instance_id': instance_id,
            'detections': [],
            'annotated_image_base64': '',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def mammography_ai_health(request):
    """
    유방촬영술 AI 서비스 헬스 체크 (Django 내장)
    
    GET /api/mri/mammography/ai/health/
    """
    try:
        # YOLO 모델 로드 확인
        model = get_yolo_model()
        
        return Response({
            'success': True,
            'service': 'mammography_ai',
            'status': 'healthy',
            'model_path': MODEL_PATH,
            'model_loaded': model is not None
        })
    except Exception as e:
        return Response({
            'success': False,
            'service': 'mammography_ai',
            'status': 'unavailable',
            'error': str(e),
            'model_path': MODEL_PATH
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
