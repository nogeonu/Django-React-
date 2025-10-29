from rest_framework import serializers
from .models import Patient, MedicalRecord

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class MedicalRecordSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.name', read_only=True)
    
    class Meta:
        model = MedicalRecord
        fields = '__all__'
        read_only_fields = ('created_at',)
