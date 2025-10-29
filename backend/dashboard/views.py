from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.db.models import Count
from datetime import datetime, timedelta
from patients.models import Patient, MedicalRecord
from medical_images.models import MedicalImage

@api_view(['GET'])
def dashboard_stats(request):
    """대시보드 통계 정보를 반환합니다."""
    
    # 전체 환자 수
    total_patients = Patient.objects.count()
    
    # 최근 30일간 등록된 환자 수
    thirty_days_ago = datetime.now() - timedelta(days=30)
    recent_patients = Patient.objects.filter(created_at__gte=thirty_days_ago).count()
    
    # 전체 의료 기록 수
    total_records = MedicalRecord.objects.count()
    
    # 최근 7일간 진료 기록 수
    seven_days_ago = datetime.now() - timedelta(days=7)
    recent_records = MedicalRecord.objects.filter(visit_date__gte=seven_days_ago).count()
    
    # 전체 의료 이미지 수
    total_images = MedicalImage.objects.count()
    
    # 최근 7일간 촬영된 이미지 수
    recent_images = MedicalImage.objects.filter(taken_date__gte=seven_days_ago).count()
    
    # 성별별 환자 분포
    gender_distribution = Patient.objects.values('gender').annotate(count=Count('id'))
    
    # 이미지 타입별 분포
    image_type_distribution = MedicalImage.objects.values('image_type').annotate(count=Count('id'))
    
    # 최근 진료 기록 (최대 5개)
    recent_medical_records = MedicalRecord.objects.select_related('patient').order_by('-visit_date')[:5]
    recent_records_data = []
    for record in recent_medical_records:
        recent_records_data.append({
            'id': record.id,
            'patient_name': record.patient.name,
            'visit_date': record.visit_date,
            'diagnosis': record.diagnosis,
        })
    
    return Response({
        'total_patients': total_patients,
        'recent_patients': recent_patients,
        'total_records': total_records,
        'recent_records': recent_records,
        'total_images': total_images,
        'recent_images': recent_images,
        'gender_distribution': list(gender_distribution),
        'image_type_distribution': list(image_type_distribution),
        'recent_medical_records': recent_records_data,
    })
