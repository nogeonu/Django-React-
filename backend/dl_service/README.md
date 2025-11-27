# 딥러닝 모델 서비스 (mosec)

## 개요
딥러닝 모델 추론을 위한 고성능 mosec 서비스입니다.
- **포트**: 5003
- **프레임워크**: mosec (Rust 기반 고성능 모델 서빙)
- **용도**: 딥러닝 모델 (PyTorch, TensorFlow 등) 추론
- **특징**: 동적 배칭, 파이프라인 스테이지, 다중 프로세스 지원

## mosec이란?

mosec은 Rust로 구축된 고성능 모델 서빙 프레임워크입니다:
- **고성능**: Rust 기반 웹 레이어로 빠른 처리 속도
- **동적 배칭**: 요청을 자동으로 배치 처리하여 GPU 활용도 극대화
- **파이프라인**: 전처리 → 추론 → 후처리를 파이프라인으로 구성
- **다중 프로세스**: CPU, GPU, I/O 작업을 효율적으로 병렬 처리
- **Python 인터페이스**: 오프라인 테스트와 동일한 코드로 서빙 가능

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install mosec torch torchvision  # 필요한 딥러닝 라이브러리 추가
```

### 2. 서비스 실행
```bash
# 개발 환경
cd backend/dl_service
python app.py

# 프로덕션 (systemd 서비스로 등록)
python app.py
```

## 아키텍처

mosec은 워커 기반 파이프라인 아키텍처를 사용합니다:

```
요청 → PreprocessWorker → InferenceWorker → PostprocessWorker → 응답
         (전처리)          (모델 추론)        (후처리)
```

각 워커는 독립적으로 실행되며, 여러 인스턴스를 생성하여 병렬 처리할 수 있습니다.

## API 엔드포인트

### 예측 API
```bash
POST http://localhost:5003/inference
Content-Type: application/json

{
  "image_data": [0.1, 0.2, ...],
  "text_data": "환자 증상 설명",
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
    "confidence": 0.85,
    "probabilities": {"class_0": 0.15, "class_1": 0.85},
    "patient_id": "P001",
    "timestamp": "2025-01-01T12:00:00",
    "model_version": "1.0.0"
  }
}
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
        timeout=30
    )
    return response.json()
```

## 모델 로드 방법

### PyTorch 모델
```python
import torch

model_path = os.path.join(current_dir, '..', 'models', 'model.pth')
model = torch.load(model_path, map_location='cpu')
model.eval()
```

### TensorFlow 모델
```python
import tensorflow as tf

model_path = os.path.join(current_dir, '..', 'models', 'model.h5')
model = tf.keras.models.load_model(model_path)
```

### ONNX 모델
```python
import onnxruntime as ort

model_path = os.path.join(current_dir, '..', 'models', 'model.onnx')
session = ort.InferenceSession(model_path)
```

## 워커 설정

`app.py`에서 워커 수를 조정할 수 있습니다:

```python
server.append_worker(PreprocessWorker, num=1)   # 전처리 워커 1개
server.append_worker(InferenceWorker, num=2)   # 추론 워커 2개 (병렬 처리)
server.append_worker(PostprocessWorker, num=1)   # 후처리 워커 1개
```

- **전처리/후처리**: CPU 작업이므로 워커 수를 적게 설정
- **추론**: GPU 작업이므로 워커 수를 늘려서 처리량 향상

## 프로덕션 배포

### systemd 서비스 등록
`/etc/systemd/system/dl-service.service`:
```ini
[Unit]
Description=Deep Learning Model Service (mosec)
After=network.target

[Service]
User=your_user
WorkingDirectory=/srv/django-react/app/backend/dl_service
Environment="PATH=/srv/django-react/app/backend/venv/bin"
ExecStart=/srv/django-react/app/backend/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable dl-service
sudo systemctl start dl-service
sudo systemctl status dl-service
```

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
