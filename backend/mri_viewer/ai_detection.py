"""
YOLO11 맘모그래피 AI 디텍션 API
"""
import os
import io
import base64
import numpy as np
from PIL import Image
import pydicom
from pathlib import Path
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

# YOLO 모델 전역 변수 (서버 시작 시 1회만 로드)
_yolo_model = None
_model_lock = None

def get_yolo_model():
    """YOLO 모델 싱글톤 로더"""
    global _yolo_model, _model_lock
    
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            import threading
            
            if _model_lock is None:
                _model_lock = threading.Lock()
            
            with _model_lock:
                if _yolo_model is None:
                    # 모델 경로 설정
                    model_path = getattr(
                        settings, 
                        'YOLO_MODEL_PATH', 
                        os.path.expanduser('~/models/yolo11_mammography/best.pt')
                    )
                    
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"YOLO model not found at {model_path}")
                    
                    print(f"Loading YOLO model from {model_path}...")
                    _yolo_model = YOLO(model_path)
                    print("✅ YOLO model loaded successfully")
        
        except ImportError:
            raise ImportError("ultralytics package not installed. Run: pip install ultralytics")
        except Exception as e:
            raise Exception(f"Failed to load YOLO model: {str(e)}")
    
    return _yolo_model


def dicom_to_image(dicom_file_path):
    """
    DICOM 파일을 PIL Image로 변환
    
    Args:
        dicom_file_path: DICOM 파일 경로
        
    Returns:
        PIL.Image: 변환된 이미지
    """
    try:
        # DICOM 파일 읽기
        ds = pydicom.dcmread(dicom_file_path)
        
        # 픽셀 데이터 가져오기
        pixel_array = ds.pixel_array
        
        # 정규화 (0-255 범위로)
        if pixel_array.max() > 255:
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(np.uint8)
        
        # Photometric Interpretation 처리
        if hasattr(ds, 'PhotometricInterpretation'):
            if ds.PhotometricInterpretation == "MONOCHROME1":
                # 반전 (MONOCHROME1은 0이 흰색)
                pixel_array = 255 - pixel_array
        
        # PIL Image로 변환
        if len(pixel_array.shape) == 2:
            # Grayscale
            image = Image.fromarray(pixel_array, mode='L')
            # RGB로 변환 (YOLO는 RGB 입력 필요)
            image = image.convert('RGB')
        else:
            # 이미 RGB
            image = Image.fromarray(pixel_array)
        
        return image
    
    except Exception as e:
        raise Exception(f"Failed to convert DICOM to image: {str(e)}")


def draw_detections(image, results, conf_threshold=0.25):
    """
    이미지에 디텍션 결과 그리기
    
    Args:
        image: PIL Image
        results: YOLO 디텍션 결과
        conf_threshold: 신뢰도 임계값
        
    Returns:
        PIL.Image: 디텍션이 그려진 이미지
    """
    from PIL import ImageDraw, ImageFont
    
    # 이미지 복사
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # 결과 파싱
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 신뢰도 확인
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # 클래스 정보
            cls = int(box.cls[0])
            class_name = result.names[cls] if hasattr(result, 'names') else f"Class {cls}"
            
            # 박스 그리기 (빨간색)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # 레이블 그리기
            label = f"{class_name} {conf:.2f}"
            
            # 텍스트 배경
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # 텍스트 크기 계산
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 배경 사각형
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill='red')
            
            # 텍스트
            draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
    
    return img_with_boxes


@api_view(['POST'])
def ai_detection(request, instance_id):
    """
    YOLO11 맘모그래피 디텍션 API
    
    Args:
        instance_id: Orthanc instance ID
        
    Returns:
        JSON response with detection results
    """
    try:
        # YOLO 모델 로드
        model = get_yolo_model()
        
        # Orthanc에서 DICOM 파일 가져오기
        from .orthanc_views import get_orthanc_client
        orthanc = get_orthanc_client()
        
        # DICOM 파일 다운로드
        dicom_data = orthanc.get_instances_id_file(instance_id)
        
        # 임시 파일로 저장
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp_file:
            tmp_file.write(dicom_data)
            tmp_path = tmp_file.name
        
        try:
            # DICOM을 이미지로 변환
            image = dicom_to_image(tmp_path)
            
            # YOLO 디텍션 수행
            print(f"Running YOLO detection on instance {instance_id}...")
            results = model(image, conf=0.25)  # 신뢰도 임계값 0.25
            
            # 결과 파싱
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                        'confidence': float(box.conf[0]),
                        'class_id': int(box.cls[0]),
                        'class_name': result.names[int(box.cls[0])] if hasattr(result, 'names') else f"Class {int(box.cls[0])}"
                    }
                    detections.append(detection)
            
            # 디텍션이 그려진 이미지 생성
            img_with_boxes = draw_detections(image, results)
            
            # 이미지를 base64로 인코딩
            buffered = io.BytesIO()
            img_with_boxes.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            print(f"✅ Detection complete: {len(detections)} objects found")
            
            return Response({
                'success': True,
                'instance_id': instance_id,
                'detections': detections,
                'detection_count': len(detections),
                'image_with_detections': f"data:image/png;base64,{img_base64}",
                'model_info': {
                    'name': 'YOLO11m Mammography',
                    'confidence_threshold': 0.25
                }
            })
        
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except FileNotFoundError as e:
        return Response({
            'success': False,
            'error': str(e),
            'message': 'YOLO model file not found. Please upload the model to the server.'
        }, status=status.HTTP_404_NOT_FOUND)
    
    except ImportError as e:
        return Response({
            'success': False,
            'error': str(e),
            'message': 'Required packages not installed. Please install ultralytics.'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    except Exception as e:
        import traceback
        print(f"❌ Detection error: {str(e)}")
        print(traceback.format_exc())
        
        return Response({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
