from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import pydicom

app = Flask(__name__)
CORS(app)

# 폐암 ML 모델 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'lung_cancer', 'ml_model', 'lung_cancer_model.pkl')
feature_path = os.path.join(current_dir, '..', 'lung_cancer', 'ml_model', 'feature_names.pkl')

# 모델 파일이 존재하는지 확인
if os.path.exists(model_path) and os.path.exists(feature_path):
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_path)
    model_loaded = True
    print(f"✅ ML 모델 로드 완료: {model_path}")
    print(f"✅ Feature names 로드 완료: {len(feature_names)}개")
else:
    model = None
    feature_names = None
    model_loaded = False
    print(f"❌ ML 모델 로드 실패: {model_path}, {feature_path}")

# YOLO 모델 로드 (유방촬영술 디텍션)
_yolo_model = None
YOLO_MODEL_PATH = os.getenv(
    'MAMMOGRAPHY_MODEL_PATH',
    '/home/shrjsdn908/models/yolo11_mammography/best.pt'
)

def get_yolo_model():
    """YOLO 모델 로드 (싱글톤 패턴)"""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model from {YOLO_MODEL_PATH}")
            _yolo_model = YOLO(YOLO_MODEL_PATH)
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
    return _yolo_model

# YOLO 모델 로드 시도
yolo_loaded = False
try:
    if os.path.exists(YOLO_MODEL_PATH):
        get_yolo_model()
        yolo_loaded = True
    else:
        print(f"⚠️  YOLO model file not found: {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"⚠️  YOLO model load failed (will skip): {e}")

@app.route('/health', methods=['GET'])
def health():
    """헬스 체크 엔드포인트"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'yolo_loaded': yolo_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    """폐암 예측 API"""
    if not model_loaded:
        return jsonify({
            'error': 'ML 모델이 로드되지 않았습니다. 모델 파일을 확인해주세요.'
        }), 500
    
    try:
        data = request.get_json()
        
        # 증상 데이터를 딕셔너리로 변환
        # 모델은 1=아니오, 2=예를 사용하므로 True를 2로, False를 1로 변환
        symptoms_dict = {
            'GENDER': 1 if data.get('gender') in ['M', '남성', '1'] else 0,
            'AGE': data.get('age', 0),
            'SMOKING': 2 if data.get('smoking') else 1,
            'YELLOW_FINGERS': 2 if data.get('yellow_fingers') else 1,
            'ANXIETY': 2 if data.get('anxiety') else 1,
            'PEER_PRESSURE': 2 if data.get('peer_pressure') else 1,
            'CHRONIC DISEASE': 2 if data.get('chronic_disease') else 1,
            'FATIGUE ': 2 if data.get('fatigue') else 1,
            'ALLERGY ': 2 if data.get('allergy') else 1,
            'WHEEZING': 2 if data.get('wheezing') else 1,
            'ALCOHOL CONSUMING': 2 if data.get('alcohol_consuming') else 1,
            'COUGHING': 2 if data.get('coughing') else 1,
            'SHORTNESS OF BREATH': 2 if data.get('shortness_of_breath') else 1,
            'SWALLOWING DIFFICULTY': 2 if data.get('swallowing_difficulty') else 1,
            'CHEST PAIN': 2 if data.get('chest_pain') else 1,
        }
        
        # DataFrame 생성 및 특성 순서 맞추기
        features = pd.DataFrame([symptoms_dict])
        features = features[feature_names]  # 특성 순서 맞추기
        
        # 예측 수행
        prediction_proba = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]
        
        # 위험도 계산
        probability_percent = float(prediction_proba[1]) * 100
        if probability_percent >= 70:
            risk_level = '높음'
            risk_message = '폐암 위험도가 높습니다. 즉시 전문의 상담을 권장합니다.'
        elif probability_percent >= 40:
            risk_level = '중간'
            risk_message = '폐암 위험도가 중간입니다. 정기적인 검진을 권장합니다.'
        else:
            risk_level = '낮음'
            risk_message = '폐암 위험도가 낮습니다. 건강한 생활 습관을 유지하세요.'
        
        return jsonify({
            'prediction': 'YES' if prediction == 1 else 'NO',
            'probability': round(probability_percent, 2),
            'risk_level': risk_level,
            'risk_message': risk_message,
            'symptoms': symptoms_dict
        }), 200
        
    except Exception as e:
        print(f"❌ 예측 중 오류: {e}")
        return jsonify({
            'error': f'예측 중 오류가 발생했습니다: {str(e)}'
        }), 500

@app.route('/mammography/detect', methods=['POST'])
def mammography_detect():
    """유방촬영술 YOLO 디텍션 API"""
    if not yolo_loaded:
        return jsonify({
            'success': False,
            'error': 'YOLO 모델이 로드되지 않았습니다.'
        }), 500
    
    try:
        data = request.get_json()
        dicom_data_base64 = data.get('dicom_data')
        confidence = float(data.get('confidence', 0.25))
        iou_threshold = float(data.get('iou_threshold', 0.45))
        
        if not dicom_data_base64:
            return jsonify({
                'success': False,
                'error': 'dicom_data가 제공되지 않았습니다.'
            }), 400
        
        # Base64 DICOM 데이터 디코딩
        dicom_data = base64.b64decode(dicom_data_base64)
        
        # DICOM을 PIL Image로 변환 (학습 시 16-bit PNG로 변환했으므로 동일한 전처리 적용)
        dicom = pydicom.dcmread(BytesIO(dicom_data))
        pixel_array = dicom.pixel_array.astype(np.float32)
        
        # Rescale Slope/Intercept 적용 (DICOM 표준)
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            pixel_array = pixel_array * float(dicom.RescaleSlope) + float(dicom.RescaleIntercept)
        
        # Window/Level 적용 (유방촬영술에서는 일반적으로 사용하지 않음)
        # 학습 시 16-bit PNG로 변환했다면, min-max 정규화를 16-bit 범위(0-65535)로 했을 가능성이 높음
        # 하지만 YOLO는 내부적으로 8-bit를 사용하므로, 학습 시에도 16-bit PNG를 0-255로 스케일링했을 가능성이 높음
        # 일단 min-max 정규화를 적용 (학습 시와 동일하게)
        if pixel_array.max() > pixel_array.min():
            # min-max 정규화: 0-65535 범위로 먼저 변환 (16-bit PNG 형식)
            pixel_array_16bit = ((pixel_array - pixel_array.min()) / 
                                (pixel_array.max() - pixel_array.min()) * 65535).astype(np.uint16)
            # YOLO는 8-bit 이미지를 입력으로 받으므로, 16-bit를 8-bit로 변환
            # 학습 시에도 16-bit PNG를 읽을 때 자동으로 0-255 범위로 스케일링되었을 가능성이 높음
            pixel_array = (pixel_array_16bit / 256).astype(np.uint8)  # 16-bit를 8-bit로 변환
        else:
            pixel_array = pixel_array.astype(np.uint8)
        
        # PhotometricInterpretation에 따라 반전
        if hasattr(dicom, 'PhotometricInterpretation'):
            if dicom.PhotometricInterpretation == 'MONOCHROME1':
                pixel_array = 255 - pixel_array
        
        # PIL Image로 변환 (8-bit grayscale -> RGB)
        # YOLO는 RGB 이미지를 입력으로 받음
        if len(pixel_array.shape) == 2:
            pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixel_array, mode='L').convert('RGB')
        else:
            pixel_array = np.clip(pixel_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixel_array)
        
        # YOLO 추론
        yolo_model = get_yolo_model()
        results = yolo_model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            device='cpu',
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
                    'bbox': box.xyxy[0].cpu().numpy().tolist(),
                    'confidence': float(box.conf[0].cpu().numpy()),
                    'class_id': int(box.cls[0].cpu().numpy()),
                    'class_name': result.names[int(box.cls[0].cpu().numpy())]
                }
                detections.append(detection)
            
            # Annotated 이미지 생성
            annotated_img = result.plot()
            annotated_pil = Image.fromarray(annotated_img[..., ::-1])
            
            buffered = BytesIO()
            annotated_pil.save(buffered, format="PNG")
            annotated_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'detection_count': len(detections),
            'annotated_image_base64': annotated_image_base64,
            'model_info': {
                'name': 'YOLO11 Mammography Detection',
                'confidence_threshold': confidence
            },
            'error': ''
        }), 200
        
    except Exception as e:
        print(f"❌ YOLO 디텍션 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'detections': [],
            'detection_count': 0,
            'annotated_image_base64': '',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # 개발 환경: 5002 포트 사용
    app.run(host='0.0.0.0', port=5002, debug=True)

