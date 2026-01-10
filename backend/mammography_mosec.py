#!/usr/bin/env python3
"""
맘모그래피 AI 분석 Mosec 서비스 (포트 5007)
ResNet50 기반 4-class 분류: Mass, Calcification, Architectural/Asymmetry, Normal
"""

import os
import io
import json
import base64
import logging
import numpy as np
import cv2
import pydicom
from PIL import Image
from typing import List, Dict

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from mosec import Server, Worker, get_logger
from mosec.mixin import MsgpackMixin

# 로깅 설정
logger = get_logger()
logger.setLevel(logging.INFO)

# GPU 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🚀 사용 디바이스: {DEVICE}")

# 모델 경로
MODEL_PATH = os.path.expanduser("~/mammography_model/resnet50_mammography_best.pth")

# 클래스 정의
CLASS_NAMES = {
    0: 'Mass',
    1: 'Calcification',
    2: 'Architectural/Asymmetry',
    3: 'Normal'
}

# ImageNet 정규화 (학습 시 사용한 값)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_resnet50_model(num_classes=4):
    """ResNet50 모델 생성 (4-class 분류)"""
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def otsu_threshold_optimized(image_16bit: np.ndarray):
    """
    OpenCV를 사용한 Otsu threshold (8비트 변환 후 적용)
    
    Returns:
        threshold: 계산된 임계값 (16비트 스케일)
        binary_image: 이진 이미지 (0 또는 65535)
    """
    img_min, img_max = image_16bit.min(), image_16bit.max()
    if img_max > img_min:
        img_8bit = ((image_16bit.astype(np.float32) - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        img_8bit = np.zeros_like(image_16bit, dtype=np.uint8)
    
    threshold_8bit, binary_8bit = cv2.threshold(img_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_16bit = int((threshold_8bit / 255.0) * (img_max - img_min) + img_min)
    binary_16bit = np.where(image_16bit > threshold_16bit, 65535, 0).astype(np.uint16)
    
    return threshold_16bit, binary_16bit


def find_contours_and_bounding_box(binary_image: np.ndarray):
    """
    윤곽선 방법을 사용하여 바운딩 박스 생성
    
    Returns:
        bounding_box: (x, y, width, height) 또는 None
    """
    binary_8bit = (binary_image / 256).astype(np.uint8)
    contours, _ = cv2.findContours(binary_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return (x, y, w, h)


def crop_image_with_bounding_box(image: np.ndarray, bounding_box, margin_ratio: float = 0.05):
    """
    바운딩 박스를 사용하여 이미지 자르기
    
    Returns:
        cropped_image: 자른 이미지
    """
    x, y, w, h = bounding_box
    img_h, img_w = image.shape[:2]
    
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    
    x_start = max(0, x - margin_x)
    y_start = max(0, y - margin_y)
    x_end = min(img_w, x + w + margin_x)
    y_end = min(img_h, y + h + margin_y)
    
    if len(image.shape) == 2:
        cropped_image = image[y_start:y_end, x_start:x_end]
    else:
        cropped_image = image[y_start:y_end, x_start:x_end, :]
    
    return cropped_image


def resize_image_preserve_aspect_ratio(image: np.ndarray, target_size=(512, 512)):
    """
    종횡비를 유지하면서 이미지를 지정된 크기로 조정
    
    Returns:
        resized_image: 크기 조정된 이미지 (512x512, 패딩 포함)
    """
    target_h, target_w = target_size
    original_h, original_w = image.shape[:2]
    
    scale_height = target_h / original_h
    scale_width = target_w / original_w
    scale_factor = min(scale_height, scale_width)
    
    new_w = int(original_w * scale_factor)
    new_h = int(original_h * scale_factor)
    
    if len(image.shape) == 2:
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 512x512로 패딩 추가 (중앙 정렬)
    if new_h < target_h or new_w < target_w:
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        
        if len(image.shape) == 2:
            resized_image = cv2.copyMakeBorder(
                resized_image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
        else:
            resized_image = cv2.copyMakeBorder(
                resized_image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
    
    return resized_image


def preprocess_dicom_image(dicom_data: bytes, target_size=(512, 512)):
    """
    DICOM 이미지 전처리 파이프라인
    
    Args:
        dicom_data: DICOM 파일 바이트 데이터
        target_size: 최종 출력 크기 (기본값: (512, 512))
    
    Returns:
        processed_image: 전처리된 이미지 (512x512, uint8, RGB 3채널)
    """
    # 1. DICOM 파일 읽기
    ds = pydicom.dcmread(io.BytesIO(dicom_data))
    pixel_array = ds.pixel_array
    
    # 2. uint16 형식으로 변환
    if pixel_array.dtype != np.uint16:
        rescale_slope = getattr(ds, 'RescaleSlope', 1.0)
        rescale_intercept = getattr(ds, 'RescaleIntercept', 0.0)
        medical_image = pixel_array * rescale_slope + rescale_intercept
        
        if medical_image.min() < 0:
            medical_image = medical_image + abs(medical_image.min())
        
        medical_image = medical_image.astype(np.uint16)
        
        img_max = medical_image.max()
        if img_max > 65535:
            medical_image = (medical_image / img_max * 65535).astype(np.uint16)
    else:
        medical_image = pixel_array.astype(np.uint16)
    
    # 3. MONOCHROME1 처리 (반전 필요 시)
    if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "MONOCHROME1":
        medical_image = 65535 - medical_image
    
    # 4. Otsu 방법을 사용한 배경 제거
    threshold, binary_image = otsu_threshold_optimized(medical_image)
    
    # 5. 윤곽선 방법을 사용한 바운딩 박스 생성
    bounding_box = find_contours_and_bounding_box(binary_image)
    
    if bounding_box is None:
        cropped_image = medical_image
    else:
        # 6. 바운딩 박스를 사용하여 이미지 자르기
        cropped_image = crop_image_with_bounding_box(medical_image, bounding_box)
    
    # 7. 종횡비 유지하며 타겟 크기로 조정
    resized_image = resize_image_preserve_aspect_ratio(cropped_image, target_size)
    
    # 8. 8비트로 변환
    image_8bit = (resized_image / 256).astype(np.uint8)
    
    # 9. RGB 3채널로 변환 (ResNet은 3채널 입력 필요)
    image_rgb = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    
    return image_rgb


class MammographyWorker(MsgpackMixin, Worker):
    """맘모그래피 AI 분석 워커"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
    
    def forward(self, data: List[Dict]) -> List[Dict]:
        """
        맘모그래피 이미지 분류 추론
        
        Args:
            data: [{"dicom_data": base64_encoded_dicom}, ...]
        
        Returns:
            [{"class_id": int, "class_name": str, "confidence": float, "probabilities": dict}, ...]
        """
        if self.model is None:
            logger.info("📦 모델 로딩 중...")
            self.model = create_resnet50_model(num_classes=4)
            
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
            
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.to(DEVICE)
            self.model.eval()
            
            # Transform 정의 (ImageNet 정규화)
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            
            logger.info(f"✅ 모델 로드 완료: {MODEL_PATH}")
        
        results = []
        
        for item in data:
            try:
                # 1. DICOM 데이터 디코딩
                dicom_base64 = item.get('dicom_data')
                if not dicom_base64:
                    raise ValueError("dicom_data가 없습니다.")
                
                dicom_bytes = base64.b64decode(dicom_base64)
                
                # 2. DICOM 전처리 (Otsu + 윤곽선 + 크롭 + 리사이즈)
                image_rgb = preprocess_dicom_image(dicom_bytes, target_size=(512, 512))
                
                # 3. PIL Image로 변환 및 Transform 적용
                image_pil = Image.fromarray(image_rgb)
                image_tensor = self.transform(image_pil).unsqueeze(0).to(DEVICE)
                
                # 4. 모델 추론
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]
                    confidence, predicted_class = torch.max(probabilities, 0)
                
                # 5. 결과 생성
                class_id = predicted_class.item()
                class_name = CLASS_NAMES[class_id]
                confidence_value = confidence.item()
                
                # 모든 클래스별 확률
                probabilities_dict = {
                    CLASS_NAMES[i]: float(probabilities[i].item())
                    for i in range(4)
                }
                
                results.append({
                    'success': True,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence_value,
                    'probabilities': probabilities_dict
                })
                
                logger.info(f"✅ 분류 완료: {class_name} (신뢰도: {confidence_value:.4f})")
                
            except Exception as e:
                logger.error(f"❌ 추론 오류: {str(e)}", exc_info=True)
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        return results


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🚀 맘모그래피 Mosec 서비스 시작 (포트 5007)")
    logger.info("="*70)
    logger.info(f"📦 모델 경로: {MODEL_PATH}")
    logger.info(f"🔧 디바이스: {DEVICE}")
    logger.info(f"📊 클래스: {list(CLASS_NAMES.values())}")
    logger.info("="*70)
    
    server = Server()
    server.append_worker(
        MammographyWorker, 
        num=1, 
        max_batch_size=8,
        max_wait_time=60  # 60초 대기
    )
    server.run()

