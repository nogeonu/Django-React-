from django.contrib import admin
from .models import Patient, Examination, MedicalImage, AIAnalysisResult


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['patient_number', 'name', 'gender', 'birth_date', 'phone', 'created_at']
    list_filter = ['gender', 'blood_type', 'created_at']
    search_fields = ['name', 'patient_number', 'phone', 'email']
    readonly_fields = ['id', 'created_at', 'updated_at']
    ordering = ['-created_at']


@admin.register(Examination)
class ExaminationAdmin(admin.ModelAdmin):
    list_display = ['patient', 'exam_type', 'body_part', 'status', 'exam_date', 'doctor_name']
    list_filter = ['exam_type', 'status', 'exam_date', 'doctor_name']
    search_fields = ['patient__name', 'patient__patient_number', 'body_part', 'doctor_name']
    readonly_fields = ['id', 'created_at']
    ordering = ['-exam_date']


@admin.register(MedicalImage)
class MedicalImageAdmin(admin.ModelAdmin):
    list_display = ['patient', 'image_type', 'body_part', 'uploaded_at', 'file_size']
    list_filter = ['image_type', 'body_part', 'uploaded_at']
    search_fields = ['patient__name', 'patient__patient_number', 'body_part', 'description']
    readonly_fields = ['id', 'uploaded_at']
    ordering = ['-uploaded_at']


@admin.register(AIAnalysisResult)
class AIAnalysisResultAdmin(admin.ModelAdmin):
    list_display = ['image', 'analysis_type', 'confidence', 'analysis_date', 'model_version']
    list_filter = ['analysis_type', 'model_version', 'analysis_date']
    search_fields = ['image__patient__name', 'image__patient__patient_number', 'findings']
    readonly_fields = ['id', 'analysis_date']
    ordering = ['-analysis_date']
