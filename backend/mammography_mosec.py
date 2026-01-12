#!/usr/bin/env python3
"""
맘모그래피 AI 분석 Mosec 서비스 (포트 5007)
ResNet50 기반 4-class 분류: Mass, Calcification, Architectural/Asymmetry, Normal
"""

import os
import io
import json
import logging
import base64
import numpy as np
import cv2
import pydicom
import requests
from PIL import Image
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from mosec import Server, Worker, get_logger

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


def generate_gradcam(model, image_tensor, target_class, original_image_shape):
    """
    Grad-CAM 히트맵 생성
    
    Args:
        model: ResNet50 모델
        image_tensor: 입력 이미지 텐서 [1, 3, H, W] (requires_grad=True)
        target_class: 타겟 클래스 인덱스
        original_image_shape: 원본 이미지 크기 (H, W)
    
    Returns:
        heatmap: Grad-CAM 히트맵 (numpy array, 0-1 normalized)
    """
    model.eval()
    
    # 마지막 convolutional layer (ResNet50의 layer4)
    target_layer = model.layer4
    
    # Gradient와 activation 저장
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        if grad_output[0] is not None:
            gradients.append(grad_output[0].cpu().data.numpy())
    
    def forward_hook(module, input, output):
        activations.append(output.cpu().data.numpy())
    
    # Hook 등록
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)
    
    try:
        # Forward pass (gradient 계산을 위해 no_grad 사용 안 함)
        output = model(image_tensor)
        
        # Target class에 대한 gradient 계산
        model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Grad-CAM 계산
        if len(gradients) == 0 or len(activations) == 0:
            logger.warning("⚠️ Grad-CAM: gradients 또는 activations가 비어있음")
            return None
        
        gradients_val = gradients[0][0]  # [C, H, W]
        activations_val = activations[0][0]  # [C, H, W]
        
        # 각 채널에 대한 가중치 계산 (gradient의 평균)
        weights = np.mean(gradients_val, axis=(1, 2))  # [C]
        
        # 가중치 합산으로 CAM 생성
        cam = np.zeros(activations_val.shape[1:], dtype=np.float32)  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations_val[i]
        
        # ReLU 적용 (양수 값만)
        cam = np.maximum(cam, 0)
        
        # 정규화
        if cam.max() > 0:
            cam = cam / (cam.max() + 1e-8)
        else:
            logger.warning("⚠️ Grad-CAM: 모든 값이 0입니다")
            return None
        
        # 원본 이미지 크기로 리사이즈
        cam_resized = cv2.resize(cam, (original_image_shape[1], original_image_shape[0]))
        
        return cam_resized
        
    except Exception as e:
        logger.error(f"❌ Grad-CAM 생성 오류: {str(e)}", exc_info=True)
        return None
    finally:
        # Hook 제거
        handle_backward.remove()
        handle_forward.remove()


def create_gradcam_overlay_on_dicom(dicom_bytes, heatmap, crop_info, alpha=0.5):
    """
    원본 DICOM 이미지와 Grad-CAM 히트맵을 오버레이 (크롭 정보 반영)
    
    Args:
        dicom_bytes: 원본 DICOM 파일 바이트
        heatmap: Grad-CAM 히트맵 (numpy array, float32, 0-1, 크롭된 이미지 기준 512x512)
        crop_info: 크롭 정보 {"bbox": (x, y, w, h), "original_shape": (H, W)}
        alpha: 히트맵 투명도 (0-1, 기본값 0.5)
    
    Returns:
        overlay_base64: 오버레이된 이미지의 base64 문자열
    """
    try:
        # DICOM 파일 읽기
        dcm = pydicom.dcmread(io.BytesIO(dicom_bytes))
        pixel_array = dcm.pixel_array
        
        # MONOCHROME1 처리 (반전)
        if hasattr(dcm, 'PhotometricInterpretation') and dcm.PhotometricInterpretation == "MONOCHROME1":
            pixel_array = pixel_array.max() - pixel_array
        
        # 정규화 (0-255)
        pixel_min = pixel_array.min()
        pixel_max = pixel_array.max()
        if pixel_max > pixel_min:
            pixel_normalized = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
        else:
            pixel_normalized = np.zeros_like(pixel_array, dtype=np.uint8)
        
        # 그레이스케일을 RGB로 변환
        if len(pixel_normalized.shape) == 2:
            dicom_rgb = cv2.cvtColor(pixel_normalized, cv2.COLOR_GRAY2RGB)
        else:
            dicom_rgb = pixel_normalized
        
        # 원본 크기의 빈 히트맵 생성
        original_h, original_w = crop_info["original_shape"]
        heatmap_full = np.zeros((original_h, original_w), dtype=np.float32)
        
        bbox = crop_info["bbox"]
        if bbox is not None:
            x, y, w, h = bbox
            
            # 히트맵을 크롭된 영역 크기로 리사이즈
            heatmap_cropped = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 원본 이미지의 크롭 영역에 히트맵 배치
            heatmap_full[y:y+h, x:x+w] = heatmap_cropped
        else:
            # 크롭이 없었다면 전체 영역에 리사이즈
            heatmap_full = cv2.resize(heatmap, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # 임계값 적용: 상위 30% 이상의 활성화 영역만 표시
        threshold = 0.3
        heatmap_thresholded = np.where(heatmap_full >= threshold, heatmap_full, 0)
        
        # 히트맵을 컬러맵으로 변환 (JET 컬러맵)
        heatmap_uint8 = (heatmap_thresholded * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # RGB로 변환 (OpenCV는 BGR 사용)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 임계값 이하 영역은 투명하게 처리 (마스크 생성)
        mask = (heatmap_thresholded > 0).astype(np.uint8)
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        
        # 원본 이미지와 히트맵 오버레이 (마스크 적용)
        overlay = dicom_rgb.copy()
        overlay = np.where(mask_3ch > 0, 
                          cv2.addWeighted(dicom_rgb, 1 - alpha, heatmap_rgb, alpha, 0),
                          dicom_rgb)
        
        # PNG로 인코딩
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return overlay_base64
        
    except Exception as e:
        logger.error(f"❌ DICOM 오버레이 생성 오류: {str(e)}", exc_info=True)
        return None


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
        crop_info: 크롭 정보 딕셔너리 {"bbox": (x, y, w, h), "original_shape": (H, W)}
    """
    # 1. DICOM 파일 읽기
    ds = pydicom.dcmread(io.BytesIO(dicom_data))
    pixel_array = ds.pixel_array
    original_shape = pixel_array.shape  # 원본 크기 저장
    
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
    
    crop_info = {
        "bbox": bounding_box,  # (x, y, w, h) 또는 None
        "original_shape": original_shape  # (H, W)
    }
    
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
    
    return image_rgb, crop_info


class MammographyWorker(Worker):
    """맘모그래피 AI 분석 워커 (Orthanc API 직접 호출)"""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.transform = None
        logger.info(f"💻 Device: {DEVICE}")
    
    def deserialize(self, data: bytes) -> dict:
        """요청 데이터 역직렬화 (Orthanc API 방식 - MRI 세그멘테이션과 동일)"""
        try:
            json_data = json.loads(data.decode('utf-8'))
            logger.info(f"📥 수신한 데이터 키: {list(json_data.keys())}")
            return json_data
        except Exception as e:
            logger.error(f"❌ 역직렬화 오류: {str(e)}")
            raise
    
    def serialize(self, data: dict) -> bytes:
        """결과 직렬화 - forward가 리스트를 반환하면 각 항목이 여기로 전달됨"""
        logger.info(f"📦 serialize 입력 타입: {type(data)}")
        
        # forward가 [{"results": [...]}]를 반환하면, 
        # Mosec이 리스트를 반복하면서 각 딕셔너리를 serialize에 전달
        if not isinstance(data, dict):
            logger.error(f"❌ serialize 예상치 못한 데이터 타입: {type(data)}, 값: {str(data)[:200]}")
            data = {"error": f"Invalid data type: {type(data)}"}
        
        json_str = json.dumps(data)
        logger.info(f"📦 JSON 길이: {len(json_str)} bytes, 키: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        return json_str.encode('utf-8')
    
    def forward(self, data) -> list:
        """
        맘모그래피 이미지 분류 추론 (Orthanc API 직접 호출)
        
        Args:
            data: dict 또는 List[dict] (Mosec 버전에 따라 다를 수 있음)
                {
                    "instance_ids": [id1, id2, id3, id4],
                    "orthanc_url": "http://localhost:8042",
                    "orthanc_auth": ["admin", "admin123"]
                }
        
        Returns:
            {"results": [...]}  # 4개 결과 포함 딕셔너리
        """
        # Mosec이 리스트로 전달할 수 있으므로 처리
        if isinstance(data, list) and len(data) > 0:
            request_data = data[0]
        elif isinstance(data, dict):
            request_data = data
        else:
            raise ValueError(f"예상치 못한 데이터 타입: {type(data)}")
        
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
        
        # Orthanc API 설정 (request_data 사용)
        instance_ids = request_data.get("instance_ids", [])
        orthanc_url = request_data.get("orthanc_url", "http://localhost:8042")
        orthanc_auth = tuple(request_data.get("orthanc_auth", ["admin", "admin123"]))
        
        logger.info(f"📥 Orthanc에서 데이터 다운로드 중: {orthanc_url}")
        logger.info(f"📊 총 {len(instance_ids)}장 이미지")
        
        results = []
        
        # Orthanc API로 각 이미지 다운로드 및 분석
        for idx, instance_id in enumerate(instance_ids):
            try:
                # Orthanc API로 DICOM 파일 다운로드
                logger.info(f"📥 DICOM 다운로드 {idx+1}/{len(instance_ids)}: {instance_id}")
                response = requests.get(
                    f"{orthanc_url}/instances/{instance_id}/file",
                    auth=orthanc_auth,
                    timeout=60
                )
                response.raise_for_status()
                dicom_bytes = response.content
                logger.info(f"✅ DICOM 다운로드 완료: {len(dicom_bytes)} bytes")
                
                # 2. DICOM 전처리 (Otsu + 윤곽선 + 크롭 + 리사이즈)
                image_rgb, crop_info = preprocess_dicom_image(dicom_bytes, target_size=(512, 512))
                original_shape = image_rgb.shape[:2]  # (H, W) 저장
                
                # 3. PIL Image로 변환 및 Transform 적용
                image_pil = Image.fromarray(image_rgb)
                image_tensor = self.transform(image_pil).unsqueeze(0).to(DEVICE)
                
                # 4. 모델 추론 (Grad-CAM을 위해 gradient 활성화)
                image_tensor_grad = image_tensor.clone().requires_grad_(True)
                outputs = self.model(image_tensor_grad)
                probabilities = torch.softmax(outputs, dim=1)[0]
                confidence, predicted_class = torch.max(probabilities, 0)
                
                # 5. Grad-CAM 생성 및 원본 DICOM에 오버레이 (병변 클래스에만)
                gradcam_overlay_base64 = None
                class_id = predicted_class.item()
                class_name = CLASS_NAMES[class_id]
                
                # Normal 클래스가 아닌 경우에만 Grad-CAM 생성
                if class_name != 'Normal':
                    try:
                        heatmap = generate_gradcam(
                            self.model, 
                            image_tensor_grad, 
                            class_id,
                            original_shape
                        )
                        
                        if heatmap is not None:
                            # 원본 DICOM 이미지에 히트맵 오버레이 (크롭 정보 사용)
                            gradcam_overlay_base64 = create_gradcam_overlay_on_dicom(
                                dicom_bytes, 
                                heatmap, 
                                crop_info,  # 크롭 정보 전달
                                alpha=0.5
                            )
                            logger.info(f"✅ Grad-CAM 오버레이 생성 완료 {idx+1}/{len(instance_ids)} - {class_name} (bbox: {crop_info['bbox']})")
                        else:
                            logger.warning(f"⚠️ Grad-CAM 생성 실패 {idx+1}/{len(instance_ids)}")
                    except Exception as e:
                        logger.error(f"❌ Grad-CAM 오버레이 생성 오류 {idx+1}/{len(instance_ids)}: {str(e)}", exc_info=True)
                else:
                    logger.info(f"ℹ️ Normal 클래스 - Grad-CAM 생성 생략 {idx+1}/{len(instance_ids)}")
                
                # 6. 결과 생성
                confidence_value = confidence.item()
                
                # 모든 클래스별 확률
                probabilities_dict = {
                    CLASS_NAMES[i]: float(probabilities[i].item())
                    for i in range(4)
                }
                
                result_item = {
                    'success': True,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence_value,
                    'probabilities': probabilities_dict
                }
                
                # Grad-CAM 오버레이가 있으면 추가
                if gradcam_overlay_base64:
                    result_item['gradcam_overlay'] = gradcam_overlay_base64
                
                results.append(result_item)
                
                logger.info(f"✅ 분류 완료 {idx+1}/{len(instance_ids)}: {class_name} (신뢰도: {confidence_value:.4f})")
                
            except Exception as e:
                logger.error(f"❌ 추론 오류 {idx+1}/{len(instance_ids)}: {str(e)}", exc_info=True)
                results.append({
                    'success': False,
                    'error': str(e)
                })
        
        # Mosec 배치 처리: 리스트 반환 (각 항목이 serialize로 전달됨)
        # 4개 이미지 결과를 하나의 딕셔너리로 묶어서 리스트에 담아 반환
        result_dict = {"results": results}
        logger.info(f"📤 forward 반환 타입: list, 길이: 1")
        logger.info(f"📤 results 길이: {len(results)}")
        return [result_dict]  # 리스트로 감싸서 반환


if __name__ == "__main__":
    logger.info("="*70)
    logger.info("🚀 맘모그래피 Mosec 서비스 시작 (포트 5007)")
    logger.info("="*70)
    logger.info(f"📦 모델 경로: {MODEL_PATH}")
    logger.info(f"🔧 디바이스: {DEVICE}")
    logger.info(f"📊 클래스: {list(CLASS_NAMES.values())}")
    logger.info("="*70)
    logger.info("⚠️  명령줄 인자로 설정: --port 5007 --timeout 120000 --max-body-size 104857600")
    logger.info("="*70)
    
    server = Server()
    server.append_worker(
        MammographyWorker, 
        num=1, 
        max_batch_size=8,
        max_wait_time=60  # 60초 대기
    )
    server.run()  # 명령줄 인자는 Mosec이 자동으로 파싱

