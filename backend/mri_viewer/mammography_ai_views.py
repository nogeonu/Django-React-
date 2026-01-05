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
        # YOLO 모델은 파일 형식이 아니라 픽셀 값에만 관심이 있으므로,
        # DICOM을 PIL Image로 변환하는 것만으로 충분합니다 (PNG 파일로 저장할 필요 없음)
        dicom = pydicom.dcmread(BytesIO(dicom_data))
        pixel_array = dicom.pixel_array.astype(np.float32)
        
        logger.info(f"DICOM pixel_array: shape={pixel_array.shape}, dtype={pixel_array.dtype}, range=[{pixel_array.min():.1f}, {pixel_array.max():.1f}]")
        
        # Rescale Slope/Intercept 적용 (DICOM 표준)
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            pixel_array = pixel_array * float(dicom.RescaleSlope) + float(dicom.RescaleIntercept)
            logger.info(f"Applied Rescale: Slope={dicom.RescaleSlope}, Intercept={dicom.RescaleIntercept}")
        
        # Window/Level 적용 (있는 경우) - 유방촬영술에서는 일반적으로 사용하지 않으므로 min-max 정규화 사용
        # 학습 데이터가 PNG였다면, 보통 min-max 정규화 또는 원본 픽셀 값을 그대로 사용했을 가능성이 높습니다
        if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
            window_center = float(dicom.WindowCenter) if isinstance(dicom.WindowCenter, (list, tuple)) else float(dicom.WindowCenter)
            window_width = float(dicom.WindowWidth) if isinstance(dicom.WindowWidth, (list, tuple)) else float(dicom.WindowWidth)
            
            window_min = window_center - window_width / 2
            window_max = window_center + window_width / 2
            
            logger.info(f"Applying Window/Level: Center={window_center}, Width={window_width}")
            # Window/Level 적용
            pixel_array = np.clip(pixel_array, window_min, window_max)
            pixel_array = ((pixel_array - window_min) / (window_max - window_min) * 255).astype(np.uint8)
        else:
            # Window/Level이 없으면 min-max 정규화 (학습 시 PNG 파일이 이 방식으로 전처리되었을 가능성이 높음)
            if pixel_array.max() > pixel_array.min():
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                logger.info(f"Applied min-max normalization: [{pixel_array.min()}, {pixel_array.max()}] -> [0, 255]")
            else:
                pixel_array = pixel_array.astype(np.uint8)
                logger.info(f"Constant pixel values, converted to uint8")
        
        # PhotometricInterpretation에 따라 반전 (MONOCHROME1인 경우)
        if hasattr(dicom, 'PhotometricInterpretation'):
            logger.info(f"PhotometricInterpretation: {dicom.PhotometricInterpretation}")
            if dicom.PhotometricInterpretation == 'MONOCHROME1':
                pixel_array = 255 - pixel_array
                logger.info("Applied inversion for MONOCHROME1")
        
        # PIL Image로 변환 (8-bit grayscale -> RGB)
        # YOLO는 RGB 이미지를 입력으로 받으므로 변환이 필요합니다
        if len(pixel_array.shape) == 2:  # Grayscale
            pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixel_array, mode='L').convert('RGB')
        else:
            pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixel_array)
        
        img_array = np.array(image)
        logger.info(f"Final PIL Image: shape={image.size}, mode={image.mode}, pixel range=[{img_array.min()}, {img_array.max()}]")
        
        # YOLO 모델 로드
        model = get_yolo_model()
        
        # YOLO 추론 실행
        confidence = float(request.data.get('confidence', 0.25))
        iou_threshold = float(request.data.get('iou_threshold', 0.45))
        
        logger.info(f"Running YOLO inference with conf={confidence}, iou={iou_threshold}, image_size={image.size}")
        results = model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            device='cpu',  # GCP VM은 CPU 사용
            verbose=True  # 디버깅을 위해 True로 변경
        )
        
        # 결과 파싱
        detections = []
        annotated_image_base64 = ""
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            logger.info(f"YOLO detected {len(boxes)} boxes (before filtering)")
            
            for box in boxes:
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                logger.info(f"  Box: class={class_name}({class_id}), conf={conf:.3f}")
                
                detection = {
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    'confidence': conf,
                    'class_id': class_id,
                    'class_name': class_name
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
        
        # 응답 구성 (프론트엔드 AIAnalysisModal 형식에 맞춤)
        response_data = {
            'success': True,
            'instance_id': instance_id,
            'detections': detections,
            'detection_count': len(detections),
            'image_with_detections': f"data:image/png;base64,{annotated_image_base64}" if annotated_image_base64 else "",
            'annotated_image_base64': annotated_image_base64,  # 하위 호환성
            'model_info': {
                'name': 'YOLO11 Mammography Detection',
                'confidence_threshold': confidence
            },
            'error': ''
        }
        
        return Response(response_data)
        
    except Exception as e:
        logger.error(f"❌ Detection failed: {str(e)}", exc_info=True)
        return Response({
            'success': False,
            'instance_id': instance_id,
            'detections': [],
            'detection_count': 0,
            'image_with_detections': '',
            'annotated_image_base64': '',
            'model_info': {
                'name': 'YOLO11 Mammography Detection',
                'confidence_threshold': 0.25
            },
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
