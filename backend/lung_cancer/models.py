from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from patients.models import Patient as CorePatient


class Patient(CorePatient):
    """patients.Patient을 의료진 플랫폼에서 재사용하기 위한 프록시 모델"""

    class Meta:
        proxy = True
        ordering = ['-created_at']

class LungRecord(models.Model):
    """폐암 검사 기록 저장"""
    patient_id = models.CharField('환자ID', max_length=10)  # Patient 테이블 ID 직접 참조
    patient_ref = models.ForeignKey(
        CorePatient,
        on_delete=models.PROTECT,
        db_column='patient_fk_id',
        related_name='lung_records_legacy',
        verbose_name='환자',
    )
    
    # 환자 기본 정보 (예측 시점 스냅샷)
    gender = models.CharField('성별', max_length=10)
    age = models.IntegerField('나이')
    
    # 검사 시점의 증상 및 생활 습관
    smoking = models.BooleanField('흡연', default=False)
    yellow_fingers = models.BooleanField('손가락 변색', default=False)
    anxiety = models.BooleanField('불안', default=False)
    peer_pressure = models.BooleanField('또래 압박', default=False)
    chronic_disease = models.BooleanField('만성 질환', default=False)
    fatigue = models.BooleanField('피로', default=False)
    allergy = models.BooleanField('알레르기', default=False)
    wheezing = models.BooleanField('쌕쌕거림', default=False)
    alcohol_consuming = models.BooleanField('음주', default=False)
    coughing = models.BooleanField('기침', default=False)
    shortness_of_breath = models.BooleanField('호흡 곤란', default=False)
    swallowing_difficulty = models.BooleanField('삼킴 곤란', default=False)
    chest_pain = models.BooleanField('가슴 통증', default=False)

    created_at = models.DateTimeField('검사일', auto_now_add=True)
    updated_at = models.DateTimeField('수정일', auto_now=True)

    class Meta:
        db_table = 'lung_record'
        managed = False  # 외부 데이터베이스 테이블
        verbose_name = '폐암 검사 데이터'
        verbose_name_plural = '폐암 검사 데이터 목록'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.patient_id} - {self.created_at.strftime('%Y-%m-%d')}"

    @property
    def patient(self):  # backward compatibility helper
        return self.patient_ref

class LungResult(models.Model):
    """검사 결과 저장"""
    lung_record = models.OneToOneField(LungRecord, on_delete=models.CASCADE, related_name='result')
    prediction = models.CharField('예측 결과', max_length=10)
    risk_score = models.DecimalField('위험 점수', max_digits=5, decimal_places=2)
    created_at = models.DateTimeField('검사일', auto_now_add=True)

    class Meta:
        db_table = 'lung_result'
        managed = False  # 외부 데이터베이스 테이블
        verbose_name = '폐암 검사 결과'
        verbose_name_plural = '폐암 검사 결과 목록'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.lung_record} - {self.prediction} ({self.risk_score}%)"


class MedicalRecord(models.Model):
    """진료기록 저장"""
    DEPARTMENT_CHOICES = [
        ('호흡기내과', '호흡기내과'),
        ('외과', '외과'),
    ]
    
    STATUS_CHOICES = [
        ('접수완료', '접수완료'),
        ('진료중', '진료중'),
        ('진료완료', '진료완료'),
    ]
    
    # 기본 정보
    id = models.AutoField(primary_key=True)
    patient_id = models.CharField('환자ID', max_length=10)
    patient_ref = models.ForeignKey(
        CorePatient,
        on_delete=models.PROTECT,
        db_column='patient_fk_id',
        related_name='medical_records_legacy',
        verbose_name='환자',
    )
    name = models.CharField('환자명', max_length=100)
    department = models.CharField('진료과', max_length=20, choices=DEPARTMENT_CHOICES)
    doctor_ref = models.ForeignKey(
        User,
        on_delete=models.PROTECT,
        db_column='doctor_fk_id',
        related_name='medical_records_as_doctor',
        verbose_name='담당의사',
        null=True,  # 기존 데이터 호환성
        blank=True,
    )
    status = models.CharField('진료상태', max_length=20, choices=STATUS_CHOICES, default='접수완료')
    notes = models.TextField('진료노트', blank=True, null=True)
    
    # 시간 관련 필드
    reception_start_time = models.DateTimeField('접수시작시간', auto_now_add=True)
    treatment_end_time = models.DateTimeField('진료끝난시간', blank=True, null=True)
    is_treatment_completed = models.BooleanField('진료완료여부', default=False)

    class Meta:
        db_table = 'medical_record'
        managed = False  # 외부 데이터베이스 테이블
        verbose_name = '진료기록'
        verbose_name_plural = '진료기록 목록'
        ordering = ['-reception_start_time']

    def __str__(self):
        return f"{self.name} - {self.department} ({self.reception_start_time.strftime('%Y-%m-%d %H:%M')})"
    
    def complete_treatment(self):
        """진료 완료 처리"""
        self.status = '진료완료'
        self.is_treatment_completed = True
        self.treatment_end_time = timezone.now()
        self.save()

    @property
    def patient(self):  # backward compatibility helper
        return self.patient_ref