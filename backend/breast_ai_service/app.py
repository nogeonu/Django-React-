"""
딥러닝 모델 서비스 - mosec
딥러닝 모델 추론을 위한 고성능 서비스 (포트 5003)
mosec은 Rust 기반의 고성능 모델 서빙 프레임워크로, 동적 배칭과 파이프라인을 지원합니다.
"""
from mosec import Worker, Server
import os
import torch
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

# 딥러닝 모델 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
# 모델 파일 경로 (환경 변수 또는 프로젝트 내부 경로)
# breast_ai_service/ml_model 디렉토리에 모델 파일 저장 (lung_cancer/ml_model 구조와 동일)
model_path = os.environ.get(
    'DL_MODEL_PATH',
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
        # 모델 로드
        self.model = None
        self.model_loaded = False
        
        if os.path.exists(model_path):
            try:
                loaded = torch.load(model_path, map_location='cpu')
                
                # 모델이 딕셔너리로 저장된 경우 처리
                if isinstance(loaded, dict):
                    if 'model_state_dict' in loaded:
                        # state_dict와 메타데이터가 있는 경우
                        self.model_data = loaded
                        self.model_state_dict = loaded['model_state_dict']
                        self.model_name = loaded.get('model_name', 'unknown')
                        self.num_classes = loaded.get('num_classes', 2)
                        self.class_names = loaded.get('class_names', ['정상', '이상'])
                        
                        # 모델 아키텍처 재구성 시도
                        try:
                            # ResNet 모델 아키텍처 시도
                            if 'resnet' in self.model_name.lower():
                                from torchvision import models
                                if '18' in self.model_name.lower():
                                    self.model = models.resnet18(pretrained=False)
                                elif '34' in self.model_name.lower():
                                    self.model = models.resnet34(pretrained=False)
                                elif '50' in self.model_name.lower():
                                    self.model = models.resnet50(pretrained=False)
                                elif '101' in self.model_name.lower():
                                    self.model = models.resnet101(pretrained=False)
                                elif '152' in self.model_name.lower():
                                    self.model = models.resnet152(pretrained=False)
                                else:
                                    # 기본값으로 ResNet50 사용
                                    self.model = models.resnet50(pretrained=False)
                                
                                # 마지막 레이어를 num_classes에 맞게 수정
                                self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
                                self.model.load_state_dict(self.model_state_dict)
                                self.model.eval()
                                self.model_loaded = True
                                print(f"✅ 딥러닝 모델 로드 완료: {self.model_name} ({self.num_classes} classes)")
                                print(f"   클래스: {self.class_names}")
                            else:
                                # 모델 아키텍처를 알 수 없는 경우
                                print(f"⚠️  모델 아키텍처를 알 수 없습니다: {self.model_name}")
                                print("⚠️  모델을 사용하려면 아키텍처 정의가 필요합니다.")
                                self.model = None
                                self.model_loaded = False
                        except Exception as e:
                            print(f"⚠️  모델 아키텍처 재구성 실패: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            self.model = None
                            self.model_loaded = False
                    elif 'model' in loaded:
                        # 모델이 'model' 키로 저장된 경우
                        self.model = loaded['model']
                        if hasattr(self.model, 'eval'):
                            self.model.eval()
                        self.model_loaded = True
                        self.class_names = loaded.get('class_names', ['정상', '이상'])
                        print(f"✅ 딥러닝 모델 로드 완료: {model_path}")
                    else:
                        # 다른 딕셔너리 형식
                        print(f"⚠️  모델 형식 확인 필요: {list(loaded.keys())}")
                        self.model = loaded
                        self.model_loaded = True
                        print(f"✅ 딥러닝 모델 로드 완료 (딕셔너리 형식): {model_path}")
                else:
                    # 모델 객체인 경우
                    self.model = loaded
                    if hasattr(self.model, 'eval'):
                        self.model.eval()
                    self.model_loaded = True
                    self.class_names = ['정상', '이상']  # 기본값
                    print(f"✅ 딥러닝 모델 로드 완료: {model_path}")
            except Exception as e:
                print(f"❌ 모델 로드 실패: {str(e)}")
                import traceback
                traceback.print_exc()
                self.model_loaded = False
        else:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            self.model_loaded = False
    
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
                - image_data: 이미지 데이터 (list 또는 numpy array)
                - image_url: 이미지 URL
                - analysis_type: 'segmentation' 또는 'classification' (기본값: 'classification')
                - patient_id: 환자 ID (선택)
                - metadata: 추가 메타데이터 (선택)
        
        Returns:
            예측 결과 딕셔너리
        """
        # analysis_type 확인 (기본값: classification)
        analysis_type = data.get('analysis_type', 'classification')
        
        # 세그멘테이션 모델은 아직 구현되지 않음
        # 임시로 분류 모델을 사용 (나중에 세그멘테이션 모델 추가 시 변경)
        if analysis_type == 'segmentation':
            # TODO: 세그멘테이션 모델 추가 시 이 부분을 세그멘테이션 모델로 변경
            # 현재는 임시로 분류 모델 사용
            logger.warning("세그멘테이션 모델이 없어 분류 모델을 사용합니다.")
            analysis_type = 'classification'  # 임시로 classification으로 변경
        
        # classification만 현재 지원
        if analysis_type != 'classification':
            return {
                'success': False,
                'error': f'지원하지 않는 분석 타입입니다: {analysis_type}',
                'data': None
            }
        
        if not self.model_loaded:
            return {
                'success': False,
                'error': '딥러닝 모델이 로드되지 않았습니다. 모델 파일을 확인해주세요.',
                'data': None
            }
        
        try:
            # 이미지 데이터 처리 (base64 또는 URL)
            image_data = data.get('image_data')
            image_url = data.get('image_url')
            
            if not image_data and not image_url:
                return {
                    'success': False,
                    'error': 'image_data 또는 image_url이 제공되지 않았습니다.',
                    'data': None
                }
            
            # 이미지 로드
            if image_data:
                # base64 디코딩
                if isinstance(image_data, str):
                    try:
                        if image_data.startswith('data:image'):
                            # data:image/png;base64,xxx 형식
                            image_data = image_data.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(BytesIO(image_bytes)).convert('RGB')
                    except Exception as e:
                        return {
                            'success': False,
                            'error': f'이미지 디코딩 실패: {str(e)}',
                            'data': None
                        }
                else:
                    # numpy array인 경우
                    try:
                        image_array = np.array(image_data)
                        if image_array.dtype != np.uint8:
                            image_array = (image_array * 255).astype(np.uint8)
                        image = Image.fromarray(image_array).convert('RGB')
                    except Exception as e:
                        return {
                            'success': False,
                            'error': f'이미지 배열 변환 실패: {str(e)}',
                            'data': None
                        }
            elif image_url:
                # URL에서 이미지 로드 (requests 필요)
                try:
                    import requests
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()  # HTTP 오류 확인
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'이미지 URL 로드 실패: {str(e)}',
                        'data': None
                    }
            else:
                return {
                    'success': False,
                    'error': 'image_data 또는 image_url이 제공되지 않았습니다.',
                    'data': None
                }
            
            # 이미지 전처리 (모델에 맞게 조정 필요)
            # 일반적인 ResNet 스타일 전처리
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 모델 입력 크기에 맞게 조정
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가
            
            # 모델 추론
            with torch.no_grad():
                if self.model is None:
                    return {
                        'success': False,
                        'error': '모델이 로드되지 않았습니다.',
                        'data': None
                    }
                
                # 모델 객체인 경우
                if hasattr(self.model, '__call__'):
                    output = self.model(input_tensor)
                else:
                    return {
                        'success': False,
                        'error': '모델이 호출 가능한 객체가 아닙니다.',
                        'data': None
                    }
                
                probabilities = torch.softmax(output, dim=1)
                prediction_idx = torch.argmax(probabilities, dim=1).item()
                confidence = float(probabilities[0][prediction_idx]) * 100
            
            # 클래스 이름 매핑
            class_names = getattr(self, 'class_names', ['Benign', 'Malignant'])
            prediction = class_names[prediction_idx] if prediction_idx < len(class_names) else f'Class_{prediction_idx}'
            
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
                class_name = class_names[i] if i < len(class_names) else f'Class_{i}'
                prob_dict[class_name] = float(prob) * 100
            
            # 발견사항 및 권장사항 생성
            findings = f"AI 분석 결과: {prediction_kr} ({prediction}) (신뢰도 {confidence:.2f}%)"
            if prediction == 'Malignant' or prediction == '악성':
                if confidence >= 80:
                    recommendations = "악성 가능성이 높습니다. 즉시 전문의 상담 및 추가 검사가 필요합니다."
                elif confidence >= 60:
                    recommendations = "악성 가능성이 있습니다. 전문의 상담을 권장합니다."
                else:
                    recommendations = "악성 가능성이 낮지만, 전문의 상담을 권장합니다."
            else:  # Benign
                if confidence >= 80:
                    recommendations = "양성으로 판단됩니다. 정기적인 검진을 권장합니다."
                elif confidence >= 60:
                    recommendations = "양성 가능성이 높습니다. 추가 검사가 필요할 수 있습니다."
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
                    'model_version': '1.0.0'
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'예측 중 오류 발생: {str(e)}',
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

