from django.contrib import admin
from .models import Patient, LungRecord, LungResult

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['patient_id', 'name', 'gender', 'age', 'phone', 'blood_type', 'created_at']
    list_filter = ['gender', 'blood_type', 'created_at']
    search_fields = ['patient_id', 'name', 'phone']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']

@admin.register(LungRecord)
class LungRecordAdmin(admin.ModelAdmin):
    list_display = ['id', 'patient_id', 'smoking', 'created_at']
    list_filter = ['smoking', 'created_at']
    search_fields = ['patient_id']
    readonly_fields = ['created_at', 'updated_at']
    ordering = ['-created_at']

@admin.register(LungResult)
class LungResultAdmin(admin.ModelAdmin):
    list_display = ['id', 'lung_record', 'prediction', 'risk_score', 'created_at']
    list_filter = ['prediction', 'created_at']
    search_fields = ['lung_record__id']
    readonly_fields = ['created_at']
    ordering = ['-created_at']