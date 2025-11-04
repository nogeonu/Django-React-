from django.contrib import admin
from .models import Patient, MedicalRecord, Appointment

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

@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = ['patient', 'title', 'appointment_date', 'appointment_type', 'status', 'created_by', 'created_at']
    list_filter = ['status', 'appointment_type', 'appointment_date', 'created_at']
    search_fields = ['patient__name', 'patient__patient_id', 'title', 'description']
    ordering = ['-appointment_date']
    date_hierarchy = 'appointment_date'
