from django.contrib import admin
from .models import Patient, MedicalRecord, PatientUser

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['patient_id', 'name', 'gender', 'birth_date', 'phone', 'created_at']
    list_filter = ['gender', 'created_at']
    search_fields = ['patient_id', 'name', 'phone']
    ordering = ['-created_at']

@admin.register(MedicalRecord)
class MedicalRecordAdmin(admin.ModelAdmin):
    list_display = ['patient', 'visit_date', 'diagnosis', 'created_at']
    list_filter = ['visit_date', 'created_at']
    search_fields = ['patient__name', 'diagnosis', 'symptoms']
    ordering = ['-visit_date']


@admin.register(PatientUser)
class PatientUserAdmin(admin.ModelAdmin):
    list_display = ['patient_id', 'account_id', 'name', 'email', 'phone', 'is_active', 'date_joined']
    list_filter = ['is_active', 'date_joined']
    search_fields = ['patient_id', 'account_id', 'name', 'email', 'phone']
    ordering = ['-date_joined']
