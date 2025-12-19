from django.db import models


class MRIStudy(models.Model):
    """MRI 검사 정보"""
    patient_id = models.CharField(max_length=100, verbose_name="환자 ID")
    study_date = models.DateField(verbose_name="검사 날짜", null=True, blank=True)
    scanner_manufacturer = models.CharField(max_length=100, verbose_name="스캐너 제조사", blank=True)
    scanner_model = models.CharField(max_length=100, verbose_name="스캐너 모델", blank=True)
    field_strength = models.FloatField(verbose_name="자기장 세기(T)", null=True, blank=True)
    
    # 환자 정보
    age = models.IntegerField(verbose_name="나이", null=True, blank=True)
    menopausal_status = models.CharField(max_length=50, verbose_name="폐경 상태", blank=True)
    tumor_subtype = models.CharField(max_length=100, verbose_name="종양 유형", blank=True)
    
    # 파일 경로
    data_directory = models.CharField(max_length=500, verbose_name="데이터 디렉토리")
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="등록일")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일")
    
    class Meta:
        db_table = 'mri_studies'
        verbose_name = 'MRI 검사'
        verbose_name_plural = 'MRI 검사 목록'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.patient_id} - {self.study_date}"

