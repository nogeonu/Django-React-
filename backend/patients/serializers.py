from rest_framework import serializers
from .models import Patient, MedicalRecord, PatientUser


class PatientUserSignupSerializer(serializers.Serializer):
    account_id = serializers.CharField(max_length=50)
    name = serializers.CharField(max_length=100)
    email = serializers.EmailField()
    phone = serializers.CharField(max_length=20)
    password = serializers.CharField(write_only=True, min_length=6)

    def validate_account_id(self, value: str):
        if PatientUser.objects.filter(account_id=value).exists():
            raise serializers.ValidationError("이미 사용 중인 계정 ID입니다.")
        return value

    def validate_email(self, value: str):
        if PatientUser.objects.filter(email=value).exists():
            raise serializers.ValidationError("이미 사용 중인 이메일입니다.")
        return value

    def validate(self, attrs):
        password = attrs.get("password", "")
        if len(password) < 6:
            raise serializers.ValidationError({"password": "비밀번호는 6자 이상이어야 합니다."})
        return attrs

    def create(self, validated_data):
        patient_id = Patient.generate_patient_id()
        patient = Patient.objects.create(
            patient_id=patient_id,
            name=validated_data["name"],
            phone=validated_data["phone"],
        )
        user = PatientUser.objects.create_user(
            account_id=validated_data["account_id"],
            email=validated_data["email"],
            password=validated_data["password"],
            name=validated_data["name"],
            patient_id=patient_id,
            phone=validated_data["phone"],
        )

        # 환자 정보와 연결
        patient.user_account = user
        patient.save(update_fields=["user_account"])

        return user


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


class PatientProfileSerializer(serializers.ModelSerializer):
    account_id = serializers.CharField(source='user_account.account_id', read_only=True)
    age = serializers.IntegerField(read_only=True)

    class Meta:
        model = Patient
        fields = [
            'patient_id',
            'name',
            'birth_date',
            'gender',
            'phone',
            'blood_type',
            'address',
            'emergency_contact',
            'medical_history',
            'allergies',
            'age',
            'account_id',
        ]
        read_only_fields = ['patient_id', 'account_id']
