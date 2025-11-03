from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import render
from django.http import JsonResponse
from django.db import models
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import requests
from .models import Patient, LungCancerPatient, LungRecord, LungResult, MedicalRecord
from .serializers import (
    PatientSerializer, 
    LungCancerPatientSerializer,
    LungRecordSerializer, 
    LungResultSerializer,
    LungCancerPredictionSerializer,
    PatientRegistrationSerializer,
    PatientUpdateSerializer,
    MedicalRecordSerializer,
    MedicalRecordCreateSerializer,
    MedicalRecordUpdateSerializer
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

# Flask ML Service URL (로컬 개발)
ML_SERVICE_URL = os.environ.get('ML_SERVICE_URL', 'http://localhost:5002')

@method_decorator(csrf_exempt, name='dispatch')
class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    
    def get_serializer_class(self):
        """요청에 따라 적절한 시리얼라이저 반환"""
        if self.action == 'update' or self.action == 'partial_update':
            return PatientUpdateSerializer
        elif self.action == 'register':
            return PatientRegistrationSerializer
        return PatientSerializer
    
    def perform_destroy(self, instance):
        """환자 삭제 시 관련 데이터도 함께 삭제"""
        from django.db import connections
        
        try:
            # hospital_db 데이터베이스 연결 사용
            with connections['default'].cursor() as cursor:
                print(f"환자 {instance.id} 삭제 시작...")
                
                # 1. LungResult 삭제
                cursor.execute("""
                    DELETE lr FROM lung_result lr
                    JOIN lung_record lrec ON lr.lung_record_id = lrec.id
                    JOIN lung_cancer_patient lcp ON lrec.lung_cancer_patient_id = lcp.id
                    WHERE lcp.patient_id = %s
                """, [instance.id])
                print(f"LungResult 삭제: {cursor.rowcount}개")
                
                # 2. LungRecord 삭제
                cursor.execute("""
                    DELETE lrec FROM lung_record lrec
                    JOIN lung_cancer_patient lcp ON lrec.lung_cancer_patient_id = lcp.id
                    WHERE lcp.patient_id = %s
                """, [instance.id])
                print(f"LungRecord 삭제: {cursor.rowcount}개")
                
                # 3. LungCancerPatient 삭제
                cursor.execute("""
                    DELETE FROM lung_cancer_patient WHERE patient_id = %s
                """, [instance.id])
                print(f"LungCancerPatient 삭제: {cursor.rowcount}개")
                
                # 4. Patient 삭제
                cursor.execute("""
                    DELETE FROM patient WHERE id = %s
                """, [instance.id])
                print(f"Patient 삭제: {cursor.rowcount}개")
            
            print(f"환자 {instance.id} 삭제 완료")
            
        except Exception as e:
            print(f"환자 삭제 중 오류: {e}")
            # 오류가 발생해도 Patient는 삭제 시도
            try:
                with connections['default'].cursor() as cursor:
                    cursor.execute("DELETE FROM patient WHERE id = %s", [instance.id])
                    print(f"환자 {instance.id} 강제 삭제 완료: {cursor.rowcount}개")
            except Exception as final_error:
                print(f"최종 삭제 실패: {final_error}")
                raise
    
    @action(detail=False, methods=['post'])
    def register(self, request):
        """환자 등록 API"""
        serializer = PatientRegistrationSerializer(data=request.data)
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
                
                # 환자 데이터 저장
                patient_data = serializer.validated_data.copy()
                patient_data['id'] = patient_id
                patient_data['age'] = age
                
                patient = Patient.objects.create(**patient_data)
                
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
        """폐암 예측 API - Flask ML Service 호출"""
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
                
                patient = Patient.objects.create(**patient_data)
                
                # 2. LungCancerPatient 테이블에 폐암 관련 정보 저장
                lung_cancer_data = {
                    'patient_id': patient.id,
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
                
                lung_cancer_patient = LungCancerPatient.objects.create(**lung_cancer_data)
                
                # 3. Flask ML Service를 통해 예측 수행
                ml_response = requests.post(
                    f'{ML_SERVICE_URL}/predict',
                    json={
                        'gender': serializer.validated_data['gender'],
                        'age': age,
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
                    },
                    timeout=10
                )
                
                if ml_response.status_code != 200:
                    return Response({
                        'error': f'ML 서비스 오류: {ml_response.json().get("error", "알 수 없는 오류")}'
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
                ml_result = ml_response.json()
                
                # 4. 예측 결과를 LungCancerPatient에 저장
                lung_cancer_patient.prediction = ml_result['prediction']
                lung_cancer_patient.prediction_probability = ml_result['probability'] / 100
                lung_cancer_patient.save()
                
                # 5. LungRecord에 검사 기록 저장
                lung_record = LungRecord.objects.create(
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
                LungResult.objects.create(
                    lung_record=lung_record,
                    prediction='양성' if lung_cancer_patient.prediction == 'YES' else '음성',
                    risk_score=lung_cancer_patient.prediction_probability * 100,
                )
                
                # 7. 결과 반환
                return Response({
                    'patient_id': patient.id,
                    'prediction': ml_result['prediction'],
                    'probability': ml_result['probability'],
                    'risk_level': ml_result['risk_level'],
                    'risk_message': ml_result['risk_message'],
                    'symptoms': ml_result['symptoms']
                }, status=status.HTTP_201_CREATED)
                
            except requests.exceptions.RequestException as e:
                return Response({
                    'error': f'ML 서비스 연결 실패: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                return Response({
                    'error': f'예측 중 오류가 발생했습니다: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['get'])
    def medical_records(self, request, pk=None):
        """특정 환자의 진료 기록 조회 API"""
        try:
            patient = self.get_object()
            # 해당 환자의 모든 진료 기록을 최신순으로 조회
            medical_records = MedicalRecord.objects.filter(
                patient_id=patient.id
            ).order_by('-reception_start_time')
            
            serializer = MedicalRecordSerializer(medical_records, many=True)
            return Response({
                'patient': PatientSerializer(patient).data,
                'medical_records': serializer.data
            })
        except Patient.DoesNotExist:
            return Response({
                'error': '환자를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'진료 기록 조회 중 오류가 발생했습니다: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LungRecordViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = LungRecord.objects.all()
    serializer_class = LungRecordSerializer

class LungResultViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = LungResult.objects.all()
    serializer_class = LungResultSerializer
    
    @action(detail=False, methods=['get'])
    def statistics(self, request):
        """폐암 예측 결과 통계"""
        try:
            results = LungResult.objects.all()
            
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
        results = LungResult.objects.all()
        
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


class MedicalRecordViewSet(viewsets.ModelViewSet):
    queryset = MedicalRecord.objects.all()
    serializer_class = MedicalRecordSerializer
    
    def get_serializer_class(self):
        if self.action == 'create':
            return MedicalRecordCreateSerializer
        elif self.action in ['update', 'partial_update']:
            return MedicalRecordUpdateSerializer
        return MedicalRecordSerializer
    
    def create(self, request):
        """진료기록 생성 API"""
        serializer = MedicalRecordCreateSerializer(data=request.data)
        if serializer.is_valid():
            try:
                patient = Patient.objects.get(id=serializer.validated_data['patient_id'])
                medical_record = MedicalRecord.objects.create(
                    patient_id=serializer.validated_data['patient_id'],
                    name=serializer.validated_data['name'],
                    department=serializer.validated_data['department'],
                    notes=serializer.validated_data.get('notes', '')
                )
                return Response({
                    'message': '진료기록이 성공적으로 생성되었습니다.',
                    'record_id': medical_record.id
                }, status=status.HTTP_201_CREATED)
            except Patient.DoesNotExist:
                return Response({
                    'error': '존재하지 않는 환자입니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({
                    'error': f'진료기록 생성 중 오류가 발생했습니다: {str(e)}'
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=True, methods=['post'])
    def complete_treatment(self, request, pk=None):
        """진료 완료 처리 API"""
        try:
            medical_record = self.get_object()
            medical_record.complete_treatment()
            return Response({
                'message': '진료가 완료되었습니다.',
                'treatment_end_time': medical_record.treatment_end_time
            }, status=status.HTTP_200_OK)
        except MedicalRecord.DoesNotExist:
            return Response({
                'error': '진료기록을 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'진료 완료 처리 중 오류가 발생했습니다: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @action(detail=False, methods=['get'])
    def by_status(self, request):
        """상태별 진료기록 조회 API"""
        status_filter = request.query_params.get('status')
        if status_filter:
            queryset = self.queryset.filter(status=status_filter)
        else:
            queryset = self.queryset
        
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def completed_today(self, request):
        """오늘 완료된 진료기록 조회 API"""
        from django.utils import timezone
        today = timezone.now().date()
        queryset = self.queryset.filter(
            is_treatment_completed=True,
            treatment_end_time__date=today
        )
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def waiting_patients(self, request):
        """대기 중인 환자 목록 조회 API (순번 기준)"""
        # 접수완료 상태의 환자들을 접수시간 순으로 정렬
        queryset = self.queryset.filter(
            status='접수완료'
        ).order_by('reception_start_time')
        
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def search_patients(self, request):
        """환자 검색 API - 호흡기내과 환자만 검색"""
        query = request.query_params.get('query', request.query_params.get('q', '')).strip()
        if not query:
            return Response({'patients': []})
        
        try:
            # 호흡기내과 진료 기록이 있는 환자들만 검색
            from django.db.models import Q
            from .models import MedicalRecord
            
            # 호흡기내과 환자 ID 목록 가져오기
            respiratory_patient_ids = MedicalRecord.objects.filter(
                department='호흡기내과'
            ).values_list('patient_id', flat=True).distinct()
            
            # 해당 환자들 중 이름으로 검색
            patients = Patient.objects.filter(
                Q(id__in=respiratory_patient_ids) &
                Q(name__icontains=query)
            ).order_by('name')[:10]  # 최대 10명만 반환, 이름순 정렬
            
            print(f"검색어: '{query}', 호흡기내과 환자 결과: {patients.count()}명")
            
            serializer = PatientSerializer(patients, many=True)
            return Response({'patients': serializer.data})
        except Exception as e:
            print(f"환자 검색 오류: {e}")
            return Response({'patients': [], 'error': str(e)})
    
    @action(detail=True, methods=['get'])
    def medical_records(self, request, pk=None):
        """특정 환자의 진료 기록 조회 API"""
        try:
            patient = self.get_object()
            # 해당 환자의 모든 진료 기록을 최신순으로 조회
            medical_records = MedicalRecord.objects.filter(
                patient_id=patient.id
            ).order_by('-reception_start_time')
            
            serializer = MedicalRecordSerializer(medical_records, many=True)
            return Response({
                'patient': PatientSerializer(patient).data,
                'medical_records': serializer.data
            })
        except Patient.DoesNotExist:
            return Response({
                'error': '환자를 찾을 수 없습니다.'
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({
                'error': f'진료 기록 조회 중 오류가 발생했습니다: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)