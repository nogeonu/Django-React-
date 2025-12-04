"""
딥러닝 모델 서비스 - mosec
딥러닝 모델 추론을 위한 고성능 서비스 (포트 5003)
mosec은 Rust 기반의 고성능 모델 서빙 프레임워크로, 동적 배칭과 파이프라인을 지원합니다.
"""
from mosec import Worker, Server
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any
from datetime import datetime
import json
import base64
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

# UNet 모델 아키텍처 정의 (원본 모델 구조에 맞춤)
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """UNet 세그멘테이션 모델 (원본 모델 구조)"""
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Downsampling path (Encoder)
        self.downs = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512)
        ])
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Upsampling path (Decoder)
        # ups는 ConvTranspose2d와 DoubleConv가 번갈아 나옴
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)
        ])
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder path with skip connections
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # 역순
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose2d
            skip = skip_connections[idx // 2]
            
            # Skip connection concatenation
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)  # DoubleConv
        
        # Final output
        return torch.sigmoid(self.final_conv(x))

# 딥러닝 모델 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
# 모델 파일 경로 (환경 변수 또는 프로젝트 내부 경로)
# breast_ai_service/ml_model 디렉토리에 모델 파일 저장

# 세그멘테이션 모델 (UNet)
segmentation_model_path = os.environ.get(
    'SEGMENTATION_MODEL_PATH',
    os.path.join(current_dir, 'ml_model', 'unet_pytorch_best.pth')
)

# 분류 모델 (ResNet 등)
classification_model_path = os.environ.get(
    'CLASSIFICATION_MODEL_PATH',
    os.path.join(current_dir, 'ml_model', 'best_breast_mri_model.pth')
)

model_loaded = False


class InferenceWorker(Worker):
    """
    딥러닝 모델 추론 워커
    mosec의 Worker를 상속받아 모델 추론을 수행합니다.
    """
    def __init__(self):
        super().__init__()
        # 두 개의 모델 로드: 세그멘테이션과 분류
        self.segmentation_model = None
        self.classification_model = None
        self.segmentation_loaded = False
        self.classification_loaded = False
        self.class_names = ['Benign', 'Malignant']  # 분류 모델용
        
        # 1. 세그멘테이션 모델 로드 (UNet)
        if os.path.exists(segmentation_model_path):
            try:
                print(f"🔄 세그멘테이션 모델 로딩 중: {segmentation_model_path}")
                seg_loaded = torch.load(segmentation_model_path, map_location='cpu')
                
                # UNet 모델 생성 (그레이스케일 입력)
                self.segmentation_model = UNet(in_channels=1, out_channels=1)
                
                # state_dict 로드
                if isinstance(seg_loaded, dict):
                    if 'model_state_dict' in seg_loaded:
                        state_dict = seg_loaded['model_state_dict']
                    elif 'state_dict' in seg_loaded:
                        state_dict = seg_loaded['state_dict']
                    else:
                        state_dict = seg_loaded
                    
                    # final_conv.bias가 있으면 제거 (코드에서는 bias=False로 설정됨)
                    if 'final_conv.bias' in state_dict:
                        state_dict = {k: v for k, v in state_dict.items() if k != 'final_conv.bias'}
                    
                    self.segmentation_model.load_state_dict(state_dict, strict=False)
                else:
                    # 모델 객체 자체인 경우
                    self.segmentation_model = seg_loaded
                
                self.segmentation_model.eval()
                self.segmentation_loaded = True
                print(f"✅ 세그멘테이션 모델 로드 완료 (UNet)")
                
            except Exception as e:
                print(f"❌ 세그멘테이션 모델 로드 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                self.segmentation_loaded = False
        else:
            print(f"⚠️  세그멘테이션 모델 파일을 찾을 수 없습니다: {segmentation_model_path}")
            self.segmentation_loaded = False
        
        # 2. 분류 모델 로드 (ResNet 등)
        if os.path.exists(classification_model_path):
            try:
                print(f"🔄 분류 모델 로딩 중: {classification_model_path}")
                cls_loaded = torch.load(classification_model_path, map_location='cpu')
                
                if isinstance(cls_loaded, dict):
                    if 'model_state_dict' in cls_loaded:
                        # state_dict와 메타데이터가 있는 경우
                        model_name = cls_loaded.get('model_name', 'unknown')
                        num_classes = cls_loaded.get('num_classes', 2)
                        self.class_names = cls_loaded.get('class_names', ['Benign', 'Malignant'])
                        
                        # ResNet 모델 아키텍처 재구성
                        if 'resnet' in model_name.lower():
                            from torchvision import models
                            if '18' in model_name.lower():
                                self.classification_model = models.resnet18(pretrained=False)
                            elif '34' in model_name.lower():
                                self.classification_model = models.resnet34(pretrained=False)
                            elif '50' in model_name.lower():
                                self.classification_model = models.resnet50(pretrained=False)
                            else:
                                self.classification_model = models.resnet50(pretrained=False)
                            
                            # 마지막 레이어 수정
                            self.classification_model.fc = nn.Linear(self.classification_model.fc.in_features, num_classes)
                            self.classification_model.load_state_dict(cls_loaded['model_state_dict'])
                            self.classification_model.eval()
                            self.classification_loaded = True
                            print(f"✅ 분류 모델 로드 완료: {model_name} ({num_classes} classes)")
                        else:
                            print(f"⚠️  알 수 없는 모델 아키텍처: {model_name}")
                            self.classification_loaded = False
                    elif 'model' in cls_loaded:
                        self.classification_model = cls_loaded['model']
                        self.classification_model.eval()
                        self.classification_loaded = True
                        print(f"✅ 분류 모델 로드 완료")
                    else:
                        self.classification_model = cls_loaded
                        if hasattr(self.classification_model, 'eval'):
                            self.classification_model.eval()
                        self.classification_loaded = True
                        print(f"✅ 분류 모델 로드 완료 (딕셔너리 형식)")
                else:
                    self.classification_model = cls_loaded
                    if hasattr(self.classification_model, 'eval'):
                        self.classification_model.eval()
                    self.classification_loaded = True
                    print(f"✅ 분류 모델 로드 완료 (모델 객체)")
                    
            except Exception as e:
                print(f"❌ 분류 모델 로드 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                self.classification_loaded = False
        else:
            print(f"⚠️  분류 모델 파일을 찾을 수 없습니다: {classification_model_path}")
            self.classification_loaded = False
    
    def deserialize(self, data: bytes) -> Dict[str, Any]:
        """요청 데이터 역직렬화 (JSON)"""
        try:
            request = json.loads(data.decode('utf-8'))
            return request
        except Exception as e:
            raise ValueError(f"요청 데이터 파싱 실패: {str(e)}")
    
    def forward(self, data):
        """
        모델 추론 수행
        
        Args:
            data: 요청 데이터 딕셔너리
                - image_data: 이미지 데이터 (base64 string)
                - image_url: 이미지 URL
                - analysis_type: 'segmentation' 또는 'classification' (기본값: 'segmentation')
                - patient_id: 환자 ID (선택)
                - metadata: 추가 메타데이터 (선택)
        
        Returns:
            예측 결과 딕셔너리
        """
        # analysis_type 확인 (기본값: segmentation)
        analysis_type = data.get('analysis_type', 'segmentation')
        
        # 세그멘테이션 분석
        if analysis_type == 'segmentation':
            return self._run_segmentation(data)
        # 분류 분석 (종양분석)
        elif analysis_type == 'classification':
            return self._run_classification(data)
        else:
            return {
                'success': False,
                'error': f'지원하지 않는 분석 타입입니다: {analysis_type}',
                'data': None
            }
    
    def _run_segmentation(self, data):
        """세그멘테이션 모델 실행 (UNet)"""
        if not self.segmentation_loaded:
            return {
                'success': False,
                'error': '세그멘테이션 모델이 로드되지 않았습니다.',
                'data': None
            }
        
        try:
            # 이미지 로드
            image = self._load_image(data)
            if image is None:
                return {
                    'success': False,
                    'error': '이미지를 로드할 수 없습니다.',
                    'data': None
                }
            
            # 원본 이미지 크기 저장
            original_size = image.size
            
            # RGB를 그레이스케일로 변환 (모델이 in_channels=1로 학습됨)
            if image.mode != 'L':
                image = image.convert('L')
            
            # 세그멘테이션 전처리 (256x256)
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            
            input_tensor = transform(image).unsqueeze(0)  # [1, 1, 256, 256]
            
            # 세그멘테이션 추론
            with torch.no_grad():
                mask = self.segmentation_model(input_tensor)  # [1, 1, 256, 256]
                mask = mask.squeeze().cpu().numpy()  # [256, 256]
            
            # 마스크를 원본 크기로 복원
            mask_resized = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            mask_resized = mask_resized.resize(original_size, Image.BILINEAR)
            
            # 마스크를 base64로 인코딩
            buffered = BytesIO()
            mask_resized.save(buffered, format="PNG")
            mask_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 종양 영역 계산
            mask_array = np.array(mask_resized) / 255.0
            tumor_area = np.sum(mask_array > 0.5)
            total_area = mask_array.size
            tumor_percentage = (tumor_area / total_area) * 100
            
            # 결과 생성
            findings = f"종양 세그멘테이션 완료. 종양 영역: {tumor_percentage:.2f}%"
            if tumor_percentage > 10:
                recommendations = "종양 영역이 감지되었습니다. 종양분석 버튼을 눌러 악성/양성 분석을 진행해주세요."
            elif tumor_percentage > 1:
                recommendations = "작은 종양 영역이 감지되었습니다. 종양분석을 권장합니다."
            else:
                recommendations = "종양 영역이 거의 감지되지 않았습니다. 재촬영 또는 전문의 상담을 권장합니다."
            
            return {
                'success': True,
                'data': {
                    'mask_image': mask_base64,
                    'tumor_percentage': round(tumor_percentage, 2),
                    'findings': findings,
                    'recommendations': recommendations,
                    'patient_id': data.get('patient_id'),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': 'UNet-1.0.0'
                }
            }
            
        except Exception as e:
            logger.error(f"세그멘테이션 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'세그멘테이션 중 오류 발생: {str(e)}',
                'data': None
            }
    
    def _load_image(self, data):
        """이미지 로드 헬퍼 메서드"""
        image_data = data.get('image_data')
        image_url = data.get('image_url')
        
        if not image_data and not image_url:
            return None
        
        try:
            if image_data:
                # base64 디코딩
                if isinstance(image_data, str):
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    return Image.open(BytesIO(image_bytes)).convert('RGB')
                else:
                    # numpy array
                    image_array = np.array(image_data)
                    if image_array.dtype != np.uint8:
                        image_array = (image_array * 255).astype(np.uint8)
                    return Image.fromarray(image_array).convert('RGB')
            elif image_url:
                import requests
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            logger.error(f"이미지 로드 실패: {str(e)}")
            return None
        
        return None
    
    def _run_classification(self, data):
        """분류 모델 실행 (악성/양성 판별)"""
        if not self.classification_loaded:
            return {
                'success': False,
                'error': '분류 모델이 로드되지 않았습니다.',
                'data': None
            }
        
        try:
            # 이미지 로드
            image = self._load_image(data)
            if image is None:
                return {
                    'success': False,
                    'error': '이미지를 로드할 수 없습니다.',
                    'data': None
                }
            
            # 분류 전처리 (224x224, ImageNet normalize)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
            
            # 분류 추론
            with torch.no_grad():
                output = self.classification_model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction_idx = torch.argmax(probabilities, dim=1).item()
                confidence = float(probabilities[0][prediction_idx]) * 100
            
            # 클래스 이름 매핑
            prediction = self.class_names[prediction_idx] if prediction_idx < len(self.class_names) else f'Class_{prediction_idx}'
            
            # 한국어 변환
            class_name_kr = {
                'Benign': '양성',
                'Malignant': '악성',
                '정상': '정상',
                '이상': '이상'
            }
            prediction_kr = class_name_kr.get(prediction, prediction)
            
            # 확률 딕셔너리 생성
            prob_dict = {}
            for i, prob in enumerate(probabilities[0]):
                class_name = self.class_names[i] if i < len(self.class_names) else f'Class_{i}'
                prob_dict[class_name] = float(prob) * 100
            
            # 발견사항 및 권장사항 생성
            findings = f"AI 종양 분석 결과: {prediction_kr} ({prediction}) (신뢰도 {confidence:.2f}%)"
            if prediction == 'Malignant' or prediction == '악성':
                if confidence >= 80:
                    recommendations = "악성 종양 가능성이 높습니다. 즉시 전문의 상담 및 추가 검사가 필요합니다."
                elif confidence >= 60:
                    recommendations = "악성 종양 가능성이 있습니다. 전문의 상담을 권장합니다."
                else:
                    recommendations = "악성 종양 가능성이 낮지만, 전문의 상담을 권장합니다."
            else:  # Benign
                if confidence >= 80:
                    recommendations = "양성 종양으로 판단됩니다. 정기적인 검진을 권장합니다."
                elif confidence >= 60:
                    recommendations = "양성 종양 가능성이 높습니다. 추가 검사가 필요할 수 있습니다."
                else:
                    recommendations = "신뢰도가 낮습니다. 재촬영 또는 추가 검사를 고려해주세요."
            
            return {
                'success': True,
                'data': {
                    'prediction': prediction_kr,
                    'prediction_en': prediction,
                    'confidence': round(confidence, 2),
                    'probabilities': prob_dict,
                    'findings': findings,
                    'recommendations': recommendations,
                    'patient_id': data.get('patient_id'),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': 'ResNet-1.0.0'
                }
            }
            
        except Exception as e:
            logger.error(f"분류 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'분류 중 오류 발생: {str(e)}',
                'data': None
            }
    
    def serialize(self, data: Dict[str, Any]) -> bytes:
        """응답 데이터 직렬화 (JSON)"""
        return json.dumps(data, ensure_ascii=False).encode('utf-8')


if __name__ == "__main__":
    # mosec 서버 설정
    # 포트 5003 사용 (환경 변수로도 설정 가능: export MOSEC_PORT=5003)
    port = int(os.environ.get('MOSEC_PORT', 5003))
    
    # mosec은 명령줄 인자로 포트를 받으므로 sys.argv 수정
    import sys
    if '--port' not in sys.argv:
        sys.argv.extend(['--port', str(port)])
    
    server = Server()
    
    # 추론 워커 추가 (여러 개 추가 시 병렬 처리)
    # num 파라미터로 워커 수 조정 (GPU 개수 또는 CPU 코어 수에 맞춤)
    server.append_worker(InferenceWorker, num=2)  # 추론 워커 2개
    
    # 서버 실행
    print(f"🚀 딥러닝 모델 서비스 시작: http://0.0.0.0:{port}")
    server.run()

