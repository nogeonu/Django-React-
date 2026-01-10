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

# MRI 모델 경로 설정
MRI_MODEL_DIR = os.getenv(
    'MRI_MODEL_DIR',
    '/home/shrjsdn908/models/mri_models'
)

@app.route('/health', methods=['GET'])
def health():
    """헬스 체크 엔드포인트"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'mri_models_available': os.path.exists(os.path.join(MRI_MODEL_DIR, 'Phase1_Segmentation_best.pth'))
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

@app.route('/mri/analyze', methods=['POST'])
def mri_analyze():
    """MRI 분석 API (pCR 예측) - 환자 ID로 분석"""
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        clinical_data = data.get('clinical_data', {})
        
        if not patient_id:
            return jsonify({
                'success': False,
                'error': 'patient_id가 제공되지 않았습니다.'
            }), 400
        
        # MRI 분석 실행 (타임아웃 600초 = 10분)
        from .mri_inference import run_mri_analysis
        result = run_mri_analysis(patient_id, clinical_data)
        
        if result.get('success', False):
            return jsonify({
                'success': True,
                'pCR_probability': result.get('pCR_probability'),
                'prediction': result.get('prediction'),
                'tumor_voxels': result.get('tumor_voxels'),
                'clinical': result.get('clinical', {})
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', '알 수 없는 오류'),
                'details': result.get('traceback')
            }), 500
            
    except Exception as e:
        import traceback
        print(f"❌ MRI 분석 API 오류: {e}", flush=True)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'MRI 분석 API 오류: {str(e)}',
            'details': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # 개발 환경: 5002 포트 사용
    app.run(host='0.0.0.0', port=5002, debug=True)

