from django.db import models
from django.utils import timezone
import os
import uuid
import re
from unicodedata import normalize

def sanitize_filename(filename):
    """
    파일명을 안전한 영어 파일명으로 변환
    - 한국어 및 특수문자 제거
    - 공백을 언더스코어로 변환
    - UUID를 추가하여 고유성 보장
    """
    # 확장자 분리
    name, ext = os.path.splitext(filename)
    
    # 한국어 및 특수문자 제거 (영문, 숫자, 언더스코어, 하이픈만 허용)
    # 먼저 유니코드 정규화 (한국어를 ASCII로 변환 시도)
    name = normalize('NFKD', name)
    # 영문, 숫자, 언더스코어, 하이픈만 남기고 나머지 제거
    name = re.sub(r'[^\w\-]', '_', name)
    # 연속된 언더스코어를 하나로
    name = re.sub(r'_+', '_', name)
    # 앞뒤 언더스코어 제거
    name = name.strip('_')
    
    # 빈 이름이면 기본값 사용
    if not name:
        name = 'image'
    
    # UUID 추가하여 고유성 보장
    unique_id = str(uuid.uuid4())[:8]
    safe_filename = f"{name}_{unique_id}{ext}"
    
    return safe_filename

def medical_image_upload_path(instance, filename):
    """
    의료 이미지 파일 저장 경로 생성
    파일명을 영어로 변환하여 저장
    경로 구조: medical_images/patient_id/YYYY/MM/DD/파일명
    """
    # 파일명 정리
    safe_filename = sanitize_filename(filename)
    
    # 환자 ID 가져오기 (instance가 저장되기 전이면 patient_id 필드에서 직접 가져옴)
    patient_id = getattr(instance, 'patient_id', None)
    if not patient_id:
        # instance가 아직 저장되지 않은 경우를 대비
        patient_id = 'unknown'
    
    # 날짜별 폴더 구조
    date_path = timezone.now().strftime('%Y/%m/%d')
    
    # 경로: medical_images/patient_id/YYYY/MM/DD/파일명
    return os.path.join('medical_images', str(patient_id), date_path, safe_filename)

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
    image_file = models.ImageField(upload_to=medical_image_upload_path, verbose_name="이미지 파일")
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
        ('BREAST_MRI_SEGMENTATION', '유방 MRI 세그멘테이션'),
        ('BREAST_MRI_CLASSIFICATION', '유방 MRI 종양분석'),
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
        max_length=30,  # BREAST_MRI_CLASSIFICATION (25자)를 수용하기 위해 30으로 증가
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
