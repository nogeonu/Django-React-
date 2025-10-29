from django.db import models
from django.contrib.auth.models import User

class Patient(models.Model):
    """환자 기본 정보 저장"""
    id = models.CharField('환자id', primary_key=True, max_length=10)  # 예: P2025001 
    name = models.CharField('환자명', max_length=100)
    birth_date = models.DateField('생년월일')
    gender = models.CharField('성별', max_length=10)
    phone = models.CharField('전화번호', max_length=20)
    address = models.TextField('주소')
    emergency_contact = models.CharField('응급연락처', max_length=20)
    blood_type = models.CharField('혈액형', max_length=5)
    age = models.IntegerField('나이')
    
    created_at = models.DateTimeField('등록일', auto_now_add=True)
    updated_at = models.DateTimeField('수정일', auto_now=True)

    class Meta:
        db_table = 'patient'
        managed = False  # 외부 데이터베이스 테이블
        verbose_name = '환자'
        verbose_name_plural = '환자 목록'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.id} - {self.name}"

class LungCancerPatient(models.Model):
    """폐암 환자 특화 정보 저장"""
    patient = models.OneToOneField(Patient, on_delete=models.CASCADE, related_name='lung_cancer_info')
    
    # 폐암 관련 증상 및 생활 습관
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
    
    # 폐암 예측 결과
    prediction = models.CharField('예측 결과', max_length=10, blank=True, null=True)
    prediction_probability = models.FloatField('예측 확률', blank=True, null=True)
    
    created_at = models.DateTimeField('등록일', auto_now_add=True)
    updated_at = models.DateTimeField('수정일', auto_now=True)

    class Meta:
        db_table = 'lung_cancer_patient'
        managed = False  # 외부 데이터베이스 테이블
        verbose_name = '폐암 환자'
        verbose_name_plural = '폐암 환자 목록'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient.name} - 폐암 정보"
    
    def get_symptoms_dict(self):
        """증상 정보를 딕셔너리로 반환 (모델 훈련용 값으로 변환)"""
        def convert_boolean(value):
            return 1 if value else 0
        
        return {
            'GENDER': 1 if self.patient.gender in ['M', '남성', '1'] else 0,
            'AGE': self.patient.age,
            'SMOKING': convert_boolean(self.smoking),
            'YELLOW_FINGERS': convert_boolean(self.yellow_fingers),
            'ANXIETY': convert_boolean(self.anxiety),
            'PEER_PRESSURE': convert_boolean(self.peer_pressure),
            'CHRONIC DISEASE': convert_boolean(self.chronic_disease),
            'FATIGUE ': convert_boolean(self.fatigue),
            'ALLERGY ': convert_boolean(self.allergy),
            'WHEEZING': convert_boolean(self.wheezing),
            'ALCOHOL CONSUMING': convert_boolean(self.alcohol_consuming),
            'COUGHING': convert_boolean(self.coughing),
            'SHORTNESS OF BREATH': convert_boolean(self.shortness_of_breath),
            'SWALLOWING DIFFICULTY': convert_boolean(self.swallowing_difficulty),
            'CHEST PAIN': convert_boolean(self.chest_pain),
        }

class LungRecord(models.Model):
    """폐암 검사 기록 저장"""
    lung_cancer_patient = models.ForeignKey(LungCancerPatient, on_delete=models.CASCADE, related_name='lung_records')
    
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
        return f"{self.lung_cancer_patient.patient.name} - {self.created_at.strftime('%Y-%m-%d')}"
    
    def get_symptoms_dict(self):
        """증상 정보를 딕셔너리로 반환 (모델 훈련용 값으로 변환)"""
        def convert_boolean(value):
            return 1 if value else 0
        
        patient = self.lung_cancer_patient.patient
        return {
            'GENDER': 1 if patient.gender in ['M', '남성', '1'] else 0,
            'AGE': patient.age,
            'SMOKING': convert_boolean(self.smoking),
            'YELLOW_FINGERS': convert_boolean(self.yellow_fingers),
            'ANXIETY': convert_boolean(self.anxiety),
            'PEER_PRESSURE': convert_boolean(self.peer_pressure),
            'CHRONIC DISEASE': convert_boolean(self.chronic_disease),
            'FATIGUE ': convert_boolean(self.fatigue),
            'ALLERGY ': convert_boolean(self.allergy),
            'WHEEZING': convert_boolean(self.wheezing),
            'ALCOHOL CONSUMING': convert_boolean(self.alcohol_consuming),
            'COUGHING': convert_boolean(self.coughing),
            'SHORTNESS OF BREATH': convert_boolean(self.shortness_of_breath),
            'SWALLOWING DIFFICULTY': convert_boolean(self.swallowing_difficulty),
            'CHEST PAIN': convert_boolean(self.chest_pain),
        }

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
        return f"{self.lung_record.patient.name} - {self.prediction} ({self.risk_score}%)"