from rest_framework import serializers
from .models import Patient, MedicalRecord, Appointment

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')
        ref_name = 'PatientsAppPatient'  # Swagger 충돌 방지

class MedicalRecordSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.name', read_only=True)
    
    class Meta:
        model = MedicalRecord
        fields = '__all__'
        read_only_fields = ('created_at',)
        ref_name = 'PatientsAppMedicalRecord'  # Swagger 충돌 방지

class AppointmentSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.name', read_only=True)
    patient_id = serializers.CharField(source='patient.patient_id', read_only=True)
    created_by_username = serializers.CharField(source='created_by.username', read_only=True)
    
    class Meta:
        model = Appointment
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')
        ref_name = 'PatientsAppAppointment'  # Swagger 충돌 방지
