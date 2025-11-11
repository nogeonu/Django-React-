from django.db import models
from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin, Group, Permission
from django.utils import timezone


class PatientUserManager(BaseUserManager):
    def create_user(self, account_id, email, password=None, **extra_fields):
        if not account_id:
            raise ValueError("계정 ID는 필수입니다.")
        if not email:
            raise ValueError("이메일은 필수입니다.")

        email = self.normalize_email(email)
        user = self.model(account_id=account_id, email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, account_id, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("슈퍼유저는 is_staff=True 여야 합니다.")
        if extra_fields.get("is_superuser") is not True:
            raise ValueError("슈퍼유저는 is_superuser=True 여야 합니다.")

        return self.create_user(account_id, email, password, **extra_fields)


class PatientUser(AbstractBaseUser, PermissionsMixin):
    account_id = models.CharField(max_length=50, unique=True, verbose_name="계정 ID")
    email = models.EmailField(unique=True, verbose_name="이메일")
    name = models.CharField(max_length=100, verbose_name="이름")
    patient_id = models.CharField(max_length=50, unique=True, verbose_name="환자 ID")
    phone = models.CharField(max_length=20, blank=True, verbose_name="전화번호")
    is_active = models.BooleanField(default=True, verbose_name="활성 여부")
    is_staff = models.BooleanField(default=False, verbose_name="스태프 여부")
    date_joined = models.DateTimeField(auto_now_add=True, verbose_name="가입일")
    groups = models.ManyToManyField(
        Group,
        related_name="patient_users",
        blank=True,
        verbose_name="소속 그룹",
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name="patient_users",
        blank=True,
        verbose_name="개별 권한",
    )

    objects = PatientUserManager()

    USERNAME_FIELD = "account_id"
    REQUIRED_FIELDS = ["email", "name", "patient_id"]

    class Meta:
        db_table = "patient_user"
        verbose_name = "환자 계정"
        verbose_name_plural = "환자 계정"

    def __str__(self):
        return f"{self.name} ({self.account_id})"


class Patient(models.Model):
    GENDER_CHOICES = [
        ('M', '남성'),
        ('F', '여성'),
    ]
    
    patient_id = models.CharField(max_length=50, unique=True, verbose_name="환자 ID")
    name = models.CharField(max_length=100, verbose_name="이름")
    birth_date = models.DateField(verbose_name="생년월일", null=True, blank=True)
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, verbose_name="성별", null=True, blank=True)
    phone = models.CharField(max_length=20, blank=True, verbose_name="전화번호")
    address = models.TextField(blank=True, verbose_name="주소")
    emergency_contact = models.CharField(max_length=100, blank=True, verbose_name="비상연락처")
    medical_history = models.TextField(blank=True, verbose_name="과거 병력")
    allergies = models.TextField(blank=True, verbose_name="알레르기")
    user_account = models.OneToOneField(
        PatientUser,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="patient_profile",
        verbose_name="연결된 환자 계정",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="등록일")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일")
    
    class Meta:
        verbose_name = "환자"
        verbose_name_plural = "환자들"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.patient_id})"

    @staticmethod
    def generate_patient_id() -> str:
        current_year = timezone.now().year
        prefix = f"P{current_year}"
        latest = (
            Patient.objects.filter(patient_id__startswith=prefix)
            .order_by('-patient_id')
            .values_list('patient_id', flat=True)
            .first()
        )
        if not latest:
            return f"{prefix}001"

        try:
            sequence = int(latest.replace(prefix, "")) + 1
        except ValueError:
            sequence = Patient.objects.filter(patient_id__startswith=prefix).count() + 1
        return f"{prefix}{sequence:03d}"


class MedicalRecord(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, related_name='medical_records', verbose_name="환자")
    visit_date = models.DateTimeField(verbose_name="방문일시")
    diagnosis = models.CharField(max_length=200, verbose_name="진단")
    symptoms = models.TextField(verbose_name="증상")
    treatment = models.TextField(verbose_name="치료내용")
    prescription = models.TextField(blank=True, verbose_name="처방전")
    doctor_notes = models.TextField(blank=True, verbose_name="의사 소견")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="등록일")
    
    class Meta:
        verbose_name = "진료 기록"
        verbose_name_plural = "진료 기록들"
        ordering = ['-visit_date']
    
    def __str__(self):
        return f"{self.patient.name} - {self.visit_date.strftime('%Y-%m-%d')}"
