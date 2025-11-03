from django.db import models

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
        db_table = 'medical_image'
        verbose_name = "의료 이미지"
        verbose_name_plural = "의료 이미지들"
        ordering = ['-taken_date']
    
    def __str__(self):
        return f"{self.patient_id} - {self.get_image_type_display()} ({self.taken_date.strftime('%Y-%m-%d')})"
