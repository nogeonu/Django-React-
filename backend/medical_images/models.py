from django.db import models
import json

class MedicalImage(models.Model):
    IMAGE_TYPE_CHOICES = [
        ('XRAY', 'X-ray'),
        ('CT', 'CT'),
        ('MRI', 'MRI'),
        ('ULTRASOUND', '초음파'),
        ('OTHER', '기타'),
    ]
    
    patient_id = models.CharField(max_length=10, verbose_name="환자 ID")
    image_type = models.CharField(max_length=20, choices=IMAGE_TYPE_CHOICES, verbose_name="이미지 타입")
    image_file = models.ImageField(upload_to='medical_images/', verbose_name="이미지 파일")
    description = models.TextField(blank=True, verbose_name="설명")
    taken_date = models.DateTimeField(verbose_name="촬영일시")
    doctor_notes = models.TextField(blank=True, verbose_name="의사 소견")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="등록일")
    
    class Meta:
        verbose_name = "의료 이미지"
        verbose_name_plural = "의료 이미지들"
        ordering = ['-taken_date']
    
    def __str__(self):
        return f"{self.patient_id} - {self.get_image_type_display()} ({self.taken_date.strftime('%Y-%m-%d')})"


class AIAnalysisResult(models.Model):
    """AI 분석 결과 모델"""
    ANALYSIS_TYPE_CHOICES = [
        ('BREAST_MRI', '유방 MRI'),
        ('LUNG_CT', '폐 CT'),
        ('XRAY', 'X-ray'),
        ('OTHER', '기타'),
    ]
    
    image = models.ForeignKey(
        MedicalImage,
        on_delete=models.CASCADE,
        related_name='analysis_results',
        verbose_name="의료 이미지"
    )
    analysis_type = models.CharField(
        max_length=20,
        choices=ANALYSIS_TYPE_CHOICES,
        default='BREAST_MRI',
        verbose_name="분석 유형"
    )
    results = models.JSONField(default=dict, verbose_name="분석 결과 (JSON)")
    confidence = models.FloatField(null=True, blank=True, verbose_name="신뢰도 (%)")
    findings = models.TextField(blank=True, verbose_name="발견사항")
    recommendations = models.TextField(blank=True, verbose_name="권장사항")
    model_version = models.CharField(max_length=50, default='1.0.0', verbose_name="모델 버전")
    analysis_date = models.DateTimeField(auto_now_add=True, verbose_name="분석일시")
    
    class Meta:
        verbose_name = "AI 분석 결과"
        verbose_name_plural = "AI 분석 결과들"
        ordering = ['-analysis_date']
    
    def __str__(self):
        return f"{self.image} - {self.get_analysis_type_display()} ({self.confidence}%)"
