from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import render
from django.http import JsonResponse
from django.db import models
from .models import Patient, LungCancerPatient, LungRecord, LungResult
from .serializers import (
    PatientSerializer, 
    LungCancerPatientSerializer,
    LungRecordSerializer, 
    LungResultSerializer,
    LungCancerPredictionSerializer,
    PatientRegistrationSerializer,
    PatientUpdateSerializer
)
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# 모델 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'ml_model', 'lung_cancer_model.pkl')
feature_path = os.path.join(current_dir, 'ml_model', 'feature_names.pkl')

# 모델 파일이 존재하는지 확인
if os.path.exists(model_path) and os.path.exists(feature_path):
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_path)
    model_loaded = True
else:
    model = None
    feature_names = None
    model_loaded = False

class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.using('hospital_db').all()
    serializer_class = PatientSerializer
    
    def get_serializer_class(self):
        """요청에 따라 적절한 시리얼라이저 반환"""
        if self.action == 'update' or self.action == 'partial_update':
            return PatientUpdateSerializer
        elif self.action == 'register':
            return PatientRegistrationSerializer
        return PatientSerializer
    
    @action(detail=False, methods=['post'])
    def register(self, request):
        """환자 등록 API"""
        serializer = PatientRegistrationSerializer(data=request.data)
        if serializer.is_valid():
            try:
                # 환자 ID 자동 생성
                from datetime import datetime
                current_year = datetime.now().year
                last_patient = Patient.objects.using('hospital_db').filter(id__startswith=f'P{current_year}').order_by('-id').first()
                if last_patient:
                    last_number = int(last_patient.id[-3:])
                    new_number = last_number + 1
                else:
                    new_number = 1
                patient_id = f'P{current_year}{new_number:03d}'
                
                # 나이 계산
                birth_date = serializer.validated_data['birth_date']
                today = datetime.now().date()
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                
                # 환자 데이터 저장
                patient_data = serializer.validated_data.copy()
                patient_data['id'] = patient_id
                patient_data['age'] = age
                
                # 증상 필드들은 기본값으로 설정
                patient_data.update({
                    'smoking': False,
                    'yellow_fingers': False,
                    'anxiety': False,
                    'peer_pressure': False,
                    'chronic_disease': False,
                    'fatigue': False,
                    'allergy': False,
                    'wheezing': False,
                    'alcohol_consuming': False,
                    'coughing': False,
                    'shortness_of_breath': False,
                    'swallowing_difficulty': False,
                    'chest_pain': False,
                })
                
                patient = Patient.objects.using('hospital_db').create(**patient_data)
                
                return Response({
                    'patient_id': patient.id,
                    'name': patient.name,
                    'age': patient.age,
                    'gender': patient.gender,
                    'message': '환자가 성공적으로 등록되었습니다.'
                }, status=status.HTTP_201_CREATED)
                
            except Exception as e:
                return Response({
                    'error': f'환자 등록 중 오류가 발생했습니다: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def predict(self, request):
        """폐암 예측 API"""
        if not model_loaded:
            return Response({
                'error': 'ML 모델이 로드되지 않았습니다. 모델 파일을 확인해주세요.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        serializer = LungCancerPredictionSerializer(data=request.data)
        if serializer.is_valid():
            try:
                # 환자 ID 자동 생성
                from datetime import datetime
                current_year = datetime.now().year
                last_patient = Patient.objects.filter(id__startswith=f'P{current_year}').order_by('-id').first()
                if last_patient:
                    last_number = int(last_patient.id[-3:])
                    new_number = last_number + 1
                else:
                    new_number = 1
                patient_id = f'P{current_year}{new_number:03d}'
                
                # 나이 계산
                birth_date = serializer.validated_data['birth_date']
                today = datetime.now().date()
                age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
                
                # 1. Patient 테이블에 기본 환자 정보 저장
                patient_data = {
                    'id': patient_id,
                    'name': serializer.validated_data['name'],
                    'birth_date': birth_date,
                    'gender': serializer.validated_data['gender'],
                    'phone': serializer.validated_data.get('phone', ''),
                    'address': serializer.validated_data.get('address', ''),
                    'emergency_contact': serializer.validated_data.get('emergency_contact', ''),
                    'blood_type': serializer.validated_data.get('blood_type', ''),
                    'age': age,
                }
                
                patient = Patient.objects.using('hospital_db').create(**patient_data)
                
                # 2. LungCancerPatient 테이블에 폐암 관련 정보 저장
                lung_cancer_data = {
                    'patient': patient,
                    'smoking': serializer.validated_data['smoking'],
                    'yellow_fingers': serializer.validated_data['yellow_fingers'],
                    'anxiety': serializer.validated_data['anxiety'],
                    'peer_pressure': serializer.validated_data['peer_pressure'],
                    'chronic_disease': serializer.validated_data['chronic_disease'],
                    'fatigue': serializer.validated_data['fatigue'],
                    'allergy': serializer.validated_data['allergy'],
                    'wheezing': serializer.validated_data['wheezing'],
                    'alcohol_consuming': serializer.validated_data['alcohol_consuming'],
                    'coughing': serializer.validated_data['coughing'],
                    'shortness_of_breath': serializer.validated_data['shortness_of_breath'],
                    'swallowing_difficulty': serializer.validated_data['swallowing_difficulty'],
                    'chest_pain': serializer.validated_data['chest_pain'],
                }
                
                lung_cancer_patient = LungCancerPatient.objects.using('hospital_db').create(**lung_cancer_data)
                
                # 3. 예측 수행을 위한 증상 데이터 준비
                symptoms_dict = lung_cancer_patient.get_symptoms_dict()
                features = pd.DataFrame([symptoms_dict])
                features = features[feature_names]  # 특성 순서 맞추기
                
                prediction_proba = model.predict_proba(features)[0]
                prediction = model.predict(features)[0]
                
                # 4. 예측 결과를 LungCancerPatient에 저장
                lung_cancer_patient.prediction = 'YES' if prediction == 1 else 'NO'
                lung_cancer_patient.prediction_probability = float(prediction_proba[1])
                lung_cancer_patient.save()
                
                # 5. LungRecord에 검사 기록 저장
                lung_record = LungRecord.objects.using('hospital_db').create(
                    lung_cancer_patient=lung_cancer_patient,
                    smoking=lung_cancer_patient.smoking,
                    yellow_fingers=lung_cancer_patient.yellow_fingers,
                    anxiety=lung_cancer_patient.anxiety,
                    peer_pressure=lung_cancer_patient.peer_pressure,
                    chronic_disease=lung_cancer_patient.chronic_disease,
                    fatigue=lung_cancer_patient.fatigue,
                    allergy=lung_cancer_patient.allergy,
                    wheezing=lung_cancer_patient.wheezing,
                    alcohol_consuming=lung_cancer_patient.alcohol_consuming,
                    coughing=lung_cancer_patient.coughing,
                    shortness_of_breath=lung_cancer_patient.shortness_of_breath,
                    swallowing_difficulty=lung_cancer_patient.swallowing_difficulty,
                    chest_pain=lung_cancer_patient.chest_pain,
                )
                
                # 6. LungResult에 검사 결과 저장
                LungResult.objects.using('hospital_db').create(
                    lung_record=lung_record,
                    prediction='양성' if lung_cancer_patient.prediction == 'YES' else '음성',
                    risk_score=lung_cancer_patient.prediction_probability * 100,
                )
                
                # 7. 위험도 계산
                probability_percent = lung_cancer_patient.prediction_probability * 100
                if probability_percent >= 70:
                    risk_level = '높음'
                    risk_message = '폐암 위험도가 높습니다. 즉시 전문의 상담을 권장합니다.'
                elif probability_percent >= 40:
                    risk_level = '중간'
                    risk_message = '폐암 위험도가 중간입니다. 정기적인 검진을 권장합니다.'
                else:
                    risk_level = '낮음'
                    risk_message = '폐암 위험도가 낮습니다. 건강한 생활 습관을 유지하세요.'
                
                return Response({
                    'patient_id': patient.id,
                    'prediction': lung_cancer_patient.prediction,
                    'probability': round(probability_percent, 2),
                    'risk_level': risk_level,
                    'risk_message': risk_message,
                    'symptoms': symptoms_dict
                }, status=status.HTTP_201_CREATED)
                
            except Exception as e:
                return Response({
                    'error': f'예측 중 오류가 발생했습니다: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LungRecordViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = LungRecord.objects.using('hospital_db').all()
    serializer_class = LungRecordSerializer

class LungResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = LungResult.objects.using('hospital_db').all()
    serializer_class = LungResultSerializer
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """폐암 예측 결과 통계"""
        try:
            results = LungResult.objects.using('hospital_db').all()
            
            total_count = results.count()
            positive_count = results.filter(prediction='양성').count()
            negative_count = results.filter(prediction='음성').count()
            
            # 성별 통계
            male_results = results.filter(lung_record__lung_cancer_patient__patient__gender__in=['M', '남성', '1'])
            female_results = results.filter(lung_record__lung_cancer_patient__patient__gender__in=['F', '여성', '0'])
            
            male_positive = male_results.filter(prediction='양성').count()
            male_negative = male_results.filter(prediction='음성').count()
            female_positive = female_results.filter(prediction='양성').count()
            female_negative = female_results.filter(prediction='음성').count()
            
            # 평균 위험도
            avg_risk_score = results.aggregate(avg_risk=models.Avg('risk_score'))['avg_risk'] or 0
            
            return Response({
                'total_patients': total_count,
                'positive_patients': positive_count,
                'negative_patients': negative_count,
                'positive_rate': round((positive_count / total_count * 100), 2) if total_count > 0 else 0,
                'gender_statistics': {
                    'male': {
                        'total': male_results.count(),
                        'positive': male_positive,
                        'negative': male_negative,
                        'positive_rate': round((male_positive / male_results.count() * 100), 2) if male_results.count() > 0 else 0
                    },
                    'female': {
                        'total': female_results.count(),
                        'positive': female_positive,
                        'negative': female_negative,
                        'positive_rate': round((female_positive / female_results.count() * 100), 2) if female_results.count() > 0 else 0
                    }
                },
                'average_risk_score': round(float(avg_risk_score), 2)
            })
        except Exception as e:
            return Response({
                'error': f'통계 조회 중 오류가 발생했습니다: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def visualization_data(request):
    """시각화 데이터 API"""
    try:
        results = LungResult.objects.using('hospital_db').all()
        
        if not results.exists():
            return JsonResponse({'error': '데이터가 없습니다.'}, status=404)
        
        # 데이터 준비
        data = []
        for result in results:
            data.append({
                'name': result.lung_record.lung_cancer_patient.patient.name,
                'gender': result.lung_record.lung_cancer_patient.patient.gender,
                'age': result.lung_record.lung_cancer_patient.patient.age,
                'prediction': 'YES' if result.prediction == '양성' else 'NO',
                'probability': float(result.risk_score) / 100,
                'created_at': result.created_at.isoformat() if result.created_at else None
            })
        
        # 통계 계산
        df = pd.DataFrame(data)
        
        # 예측 결과 분포
        prediction_counts = df['prediction'].value_counts()
        
        # 성별 분포
        df['gender_label'] = df['gender'].apply(lambda x: '남성' if x in ['M', '남성', '1'] else '여성')
        gender_prediction = pd.crosstab(df['gender_label'], df['prediction'])
        
        # 연령대별 분포
        df['age_decade'] = (df['age'] // 10) * 10
        age_prediction = pd.crosstab(df['age_decade'], df['prediction'])
        
        return JsonResponse({
            'prediction_distribution': prediction_counts.to_dict(),
            'gender_distribution': gender_prediction.to_dict(),
            'age_distribution': age_prediction.to_dict(),
            'total_patients': len(df),
            'average_age': round(df['age'].mean(), 1),
            'average_risk': round(df['probability'].mean() * 100, 2)
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'시각화 데이터 생성 중 오류가 발생했습니다: {str(e)}'
        }, status=500)