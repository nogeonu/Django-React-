from rest_framework import serializers
from .models import Patient, LungCancerPatient, LungRecord, LungResult

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = '__all__'
        read_only_fields = ['created_at', 'updated_at']

class LungCancerPatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = LungCancerPatient
        fields = '__all__'
        read_only_fields = ['created_at', 'updated_at']

class LungRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = LungRecord
        fields = '__all__'

class LungResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = LungResult
        fields = '__all__'

class PatientRegistrationSerializer(serializers.Serializer):
    """환자 등록을 위한 시리얼라이저"""
    name = serializers.CharField(max_length=100)
    birth_date = serializers.DateField()
    gender = serializers.CharField(max_length=10)
    phone = serializers.CharField(max_length=20, required=False, allow_blank=True)
    address = serializers.CharField(required=False, allow_blank=True)
    emergency_contact = serializers.CharField(max_length=20, required=False, allow_blank=True)
    blood_type = serializers.CharField(max_length=5, required=False, allow_blank=True)
    
    def validate_gender(self, value):
        if value not in ['M', 'F', '남성', '여성', '1', '0']:
            raise serializers.ValidationError("성별은 M/F, 남성/여성, 1/0 중 하나여야 합니다.")
        return value

class PatientUpdateSerializer(serializers.ModelSerializer):
    """환자 수정을 위한 시리얼라이저"""
    class Meta:
        model = Patient
        fields = ['name', 'birth_date', 'gender', 'phone', 'address', 'emergency_contact', 'blood_type']
        read_only_fields = ['id', 'age', 'created_at', 'updated_at']
    
    def validate_gender(self, value):
        if value not in ['M', 'F', '남성', '여성', '1', '0']:
            raise serializers.ValidationError("성별은 M/F, 남성/여성, 1/0 중 하나여야 합니다.")
        return value
    
    def update(self, instance, validated_data):
        # 생년월일이 변경되면 나이도 다시 계산
        if 'birth_date' in validated_data:
            from datetime import datetime
            birth_date = validated_data['birth_date']
            today = datetime.now().date()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            validated_data['age'] = age
        
        return super().update(instance, validated_data)

class LungCancerPredictionSerializer(serializers.Serializer):
    """폐암 예측을 위한 입력 데이터 시리얼라이저"""
    name = serializers.CharField(max_length=100, required=False, allow_blank=True)
    birth_date = serializers.DateField()
    gender = serializers.CharField(max_length=10)
    phone = serializers.CharField(max_length=20, required=False, allow_blank=True)
    address = serializers.CharField(required=False, allow_blank=True)
    emergency_contact = serializers.CharField(max_length=20, required=False, allow_blank=True)
    blood_type = serializers.CharField(max_length=5, required=False, allow_blank=True)
    age = serializers.IntegerField()
    smoking = serializers.BooleanField()
    yellow_fingers = serializers.BooleanField()
    anxiety = serializers.BooleanField()
    peer_pressure = serializers.BooleanField()
    chronic_disease = serializers.BooleanField()
    fatigue = serializers.BooleanField()
    allergy = serializers.BooleanField()
    wheezing = serializers.BooleanField()
    alcohol_consuming = serializers.BooleanField()
    coughing = serializers.BooleanField()
    shortness_of_breath = serializers.BooleanField()
    swallowing_difficulty = serializers.BooleanField()
    chest_pain = serializers.BooleanField()
    
    def validate_gender(self, value):
        if value not in ['M', 'F', '남성', '여성', '1', '0']:
            raise serializers.ValidationError("성별은 M/F, 남성/여성, 1/0 중 하나여야 합니다.")
        return value
    
    def validate_age(self, value):
        if value < 0 or value > 120:
            raise serializers.ValidationError("나이는 0-120 사이의 값이어야 합니다.")
        return value