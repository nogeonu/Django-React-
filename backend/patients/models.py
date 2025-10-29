from django.db import models
from django.contrib.auth.models import User

class Patient(models.Model):
    GENDER_CHOICES = [
        ('M', '남성'),
        ('F', '여성'),
    ]
    
    patient_id = models.CharField(max_length=50, unique=True, verbose_name="환자 ID")
    name = models.CharField(max_length=100, verbose_name="이름")
    birth_date = models.DateField(verbose_name="생년월일")
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, verbose_name="성별")
    phone = models.CharField(max_length=20, blank=True, verbose_name="전화번호")
    address = models.TextField(blank=True, verbose_name="주소")
    emergency_contact = models.CharField(max_length=100, blank=True, verbose_name="비상연락처")
    medical_history = models.TextField(blank=True, verbose_name="과거 병력")
    allergies = models.TextField(blank=True, verbose_name="알레르기")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="등록일")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일")
    
    class Meta:
        verbose_name = "환자"
        verbose_name_plural = "환자들"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.patient_id})"

class MedicalRecord(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='medical_records', verbose_name="환자")
    visit_date = models.DateTimeField(verbose_name="방문일시")
    diagnosis = models.CharField(max_length=200, verbose_name="진단")
    symptoms = models.TextField(verbose_name="증상")
    treatment = models.TextField(verbose_name="치료내용")
    prescription = models.TextField(blank=True, verbose_name="처방전")
    doctor_notes = models.TextField(blank=True, verbose_name="의사 소견")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="등록일")
    
    class Meta:
        verbose_name = "진료 기록"
        verbose_name_plural = "진료 기록들"
        ordering = ['-visit_date']
    
    def __str__(self):
        return f"{self.patient.name} - {self.visit_date.strftime('%Y-%m-%d')}"
