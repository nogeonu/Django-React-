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
import cv2

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
        
        # DICOM을 PIL Image로 변환 (학습 시 전처리 파이프라인과 동일하게 적용)
        # 학습 시 전처리: DICOM → Rescale → MONOCHROME1 반전 → 배경제거 → CLAHE & Windowing → 16-bit PNG
        dicom = pydicom.dcmread(BytesIO(dicom_data))
        pixel_array = dicom.pixel_array.astype(np.float32)
        
        print(f"DICOM pixel_array: shape={pixel_array.shape}, dtype={pixel_array.dtype}, range=[{pixel_array.min():.1f}, {pixel_array.max():.1f}]")
        
        # 1. Rescale Slope/Intercept 적용 (DICOM 표준)
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            pixel_array = pixel_array * float(dicom.RescaleSlope) + float(dicom.RescaleIntercept)
            print(f"Applied Rescale: Slope={dicom.RescaleSlope}, Intercept={dicom.RescaleIntercept}, new range=[{pixel_array.min():.1f}, {pixel_array.max():.1f}]")
        
        # 2. PhotometricInterpretation에 따라 반전 (MONOCHROME1인 경우)
        if hasattr(dicom, 'PhotometricInterpretation'):
            if dicom.PhotometricInterpretation == 'MONOCHROME1':
                pixel_array = pixel_array.max() - pixel_array + pixel_array.min()  # 반전
                print(f"Applied inversion for MONOCHROME1")
        
        # 3. 16-bit 범위로 정규화 (학습 시 16-bit PNG 저장과 동일)
        if pixel_array.max() > pixel_array.min():
            pixel_array_16bit = ((pixel_array - pixel_array.min()) / 
                                (pixel_array.max() - pixel_array.min()) * 65535).astype(np.uint16)
        else:
            pixel_array_16bit = pixel_array.astype(np.uint16)
        
        # 4. 배경 제거 (Otsu threshold) - 선택사항이지만 학습 시 사용했으므로 적용
        # OpenCV는 uint16을 직접 지원하므로 16-bit로 처리
        pixel_array_16bit_cv = pixel_array_16bit.astype(np.uint16)
        
        # Otsu threshold 적용 (배경 제거)
        # 16-bit 이미지에 대해 Otsu 적용을 위해 8-bit로 변환 후 적용
        pixel_array_8bit_temp = (pixel_array_16bit_cv / 256).astype(np.uint8)
        _, binary_mask = cv2.threshold(pixel_array_8bit_temp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 마스크를 16-bit 범위로 확장
        binary_mask_16bit = (binary_mask.astype(np.uint16) * 256)
        
        # 배경을 0으로 설정 (선택사항 - 모델이 자동 처리 가능하므로 주석 처리 가능)
        # pixel_array_16bit_cv = np.where(binary_mask_16bit > 0, pixel_array_16bit_cv, 0)
        
        # 5. CLAHE & Windowing (대비 향상) - 선택사항이지만 학습 시 사용
        # 16-bit 이미지에 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # CLAHE는 8-bit 또는 16-bit 입력을 받지만, 16-bit는 제한적이므로 8-bit로 변환 후 적용
        pixel_array_8bit_for_clahe = (pixel_array_16bit_cv / 256).astype(np.uint8)
        pixel_array_enhanced = clahe.apply(pixel_array_8bit_for_clahe)
        
        # 6. 16-bit PNG 형식으로 저장했으므로, YOLO 입력을 위해 8-bit로 변환
        # 학습 시 16-bit PNG를 읽을 때 PIL/OpenCV가 자동으로 0-255로 스케일링하므로 동일하게 처리
        pixel_array_final = pixel_array_enhanced.astype(np.uint8)
        
        print(f"Final processed image: range=[{pixel_array_final.min()}, {pixel_array_final.max()}]")
        
        # 7. PIL Image로 변환 (grayscale -> RGB)
        # YOLO는 RGB 이미지를 입력으로 받음
        image = Image.fromarray(pixel_array_final, mode='L').convert('RGB')
        
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

