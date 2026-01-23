# 딥러닝 모델 서비스 (mosec)

## 개요
딥러닝 모델 추론을 위한 고성능 mosec 서비스입니다.
- **포트**: 5003
- **프레임워크**: mosec (Rust 기반 고성능 모델 서빙)
- **용도**: 딥러닝 모델 (PyTorch, TensorFlow 등) 추론
- **특징**: 동적 배칭, 파이프라인 스테이지, 다중 프로세스 지원

## 로컬 개발 환경

### 1. 설치 및 실행

#### 방법 1: 스크립트 사용 (권장)
```bash
cd backend/breast_ai_service
./start_local.sh
```

#### 방법 2: 수동 실행
```bash
cd backend/breast_ai_service

# mosec 설치 (처음 한 번만)
pip3 install mosec torch torchvision

# 환경 변수 설정 (선택사항)
export MOSEC_PORT=5003
export DL_MODEL_PATH=$(pwd)/ml_model/best_breast_mri_model.pth

# 서비스 실행
python3 app.py
```

### 2. 확인
서비스가 정상적으로 실행되면 다음 메시지가 표시됩니다:
```
✅ 딥러닝 모델 로드 완료: ...
🚀 딥러닝 모델 서비스 시작: http://0.0.0.0:5003
```

## 프로덕션 환경 (GCP)

### 자동 배포
GitHub Actions를 통해 자동으로 배포되며, systemd 서비스로 등록됩니다.

### 수동 설정 (필요시)

#### 1. SSH로 GCP 서버 접속
```bash
ssh -i ~/.ssh/your_key user@your-server-ip
```

#### 2. systemd 서비스 확인
```bash
# 서비스 상태 확인
sudo systemctl status breast-ai-service

# 서비스 시작
sudo systemctl start breast-ai-service

# 서비스 재시작
sudo systemctl restart breast-ai-service

# 로그 확인
sudo journalctl -u breast-ai-service -f
```

#### 3. 수동 실행 (디버깅용)
```bash
cd /srv/django-react/app/backend/breast_ai_service
source ../.venv/bin/activate
export MOSEC_PORT=5003
python app.py
```

## API 엔드포인트

### 예측 API
```bash
POST http://localhost:5003/inference
Content-Type: application/json

{
  "image_data": "base64_encoded_image",
  "image_url": "http://...",
  "patient_id": "P001",
  "metadata": {}
}
```

### 응답 형식
```json
{
  "success": true,
  "data": {
    "prediction": "예측 결과",
    "confidence": 85.5,
    "probabilities": {"정상": 15.0, "이상": 85.0},
    "findings": "AI 분석 결과: 이상 (신뢰도 85.50%)",
    "recommendations": "높은 신뢰도로 진단되었습니다. 전문의 상담을 권장합니다.",
    "patient_id": "P001",
    "timestamp": "2025-01-01T12:00:00",
    "model_version": "1.0.0"
  }
}
```

## 문제 해결

### 1. 포트가 이미 사용 중인 경우
```bash
# 포트 사용 확인
lsof -i :5003

# 다른 포트 사용
export MOSEC_PORT=5004
python3 app.py
```

### 2. 모델 파일을 찾을 수 없는 경우
```bash
# 모델 경로 확인
ls -lh backend/breast_ai_service/ml_model/

# 환경 변수로 경로 지정
export DL_MODEL_PATH=/path/to/your/model.pth
python3 app.py
```

### 3. mosec이 설치되지 않은 경우
```bash
pip3 install mosec torch torchvision
```

### 4. GCP에서 서비스가 시작되지 않는 경우
```bash
# 로그 확인
sudo journalctl -u breast-ai-service -n 50

# 서비스 재시작
sudo systemctl restart breast-ai-service
```

## Django에서 호출하기

```python
# settings.py 또는 views.py
DL_SERVICE_URL = os.environ.get('DL_SERVICE_URL', 'http://localhost:5003')

# views.py에서 호출 예시
import requests

def predict_with_dl_model(data):
    response = requests.post(
        f'{DL_SERVICE_URL}/inference',
        json=data,
        timeout=60
    )
    return response.json()
```

## 모델 로드 방법

### PyTorch 모델
```python
import torch

model_path = os.path.join(current_dir, 'ml_model', 'model.pth')
model = torch.load(model_path, map_location='cpu')
model.eval()
```

### TensorFlow 모델
```python
import tensorflow as tf

model_path = os.path.join(current_dir, 'ml_model', 'model.h5')
model = tf.keras.models.load_model(model_path)
```

### ONNX 모델
```python
import onnxruntime as ort

model_path = os.path.join(current_dir, 'ml_model', 'model.onnx')
session = ort.InferenceSession(model_path)
```

## 워커 설정

`app.py`에서 워커 수를 조정할 수 있습니다:

```python
server.append_worker(InferenceWorker, num=2)  # 추론 워커 2개
```

- **CPU만 있는 경우**: CPU 코어 수에 맞춰 워커 수 조정
- **GPU가 있는 경우**: GPU 개수에 맞춰 워커 수 조정

## 성능 최적화

### 1. 워커 수 조정
- GPU가 있는 경우: 추론 워커 수를 GPU 개수에 맞춤
- CPU만 있는 경우: CPU 코어 수에 맞춰 워커 수 조정

### 2. 배치 크기
mosec은 자동으로 배치 처리를 수행하지만, 필요시 수동으로 조정 가능

### 3. 모델 최적화
- ONNX 변환: PyTorch/TensorFlow 모델을 ONNX로 변환하여 성능 향상
- 양자화: INT8 양자화로 추론 속도 향상
- TensorRT: NVIDIA GPU 사용 시 TensorRT 최적화

## ML 서비스와의 차이점

| 항목 | ML Service (Flask) | DL Service (mosec) |
|------|-------------------|-------------------|
| 포트 | 5002 | 5003 |
| 프레임워크 | Flask | mosec |
| 모델 타입 | scikit-learn (PKL) | PyTorch/TensorFlow/ONNX |
| 성능 | 일반 | 매우 우수 (Rust 기반) |
| 배칭 | 수동 | 자동 (동적 배칭) |
| 파이프라인 | 단일 프로세스 | 다중 워커 파이프라인 |
| 확장성 | 제한적 | 우수 (다중 프로세스) |

## 참고 자료

- [mosec 공식 문서](https://mosec.readthedocs.io/)
- [mosec GitHub](https://github.com/mosecorg/mosec)
