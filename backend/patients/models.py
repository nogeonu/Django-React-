import uuid
from django.conf import settings
from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.contrib.auth.models import PermissionsMixin, Group, Permission
from django.db import models
from django.utils import timezone
from eventeye.doctor_utils import get_department, get_doctor_id


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
    blood_type = models.CharField(max_length=3, blank=True, null=True, verbose_name="혈액형")
    address = models.TextField(blank=True, verbose_name="주소")
    emergency_contact = models.CharField(max_length=100, blank=True, verbose_name="비상연락처")
    medical_history = models.TextField(blank=True, verbose_name="과거 병력")
    allergies = models.TextField(blank=True, verbose_name="알레르기")
    age = models.PositiveIntegerField(null=True, blank=True, verbose_name="나이")
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

    def update_age(self):
        if self.birth_date:
            today = timezone.now().date()
            years = today.year - self.birth_date.year
            if (today.month, today.day) < (self.birth_date.month, self.birth_date.day):
                years -= 1
            self.age = max(years, 0)
        else:
            self.age = None

    def save(self, *args, **kwargs):
        self.update_age()
        super().save(*args, **kwargs)

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


class Appointment(models.Model):
    """의료 예약 정보"""

    STATUS_SCHEDULED = "scheduled"
    STATUS_COMPLETED = "completed"
    STATUS_CANCELLED = "cancelled"
    STATUS_CHOICES = [
        (STATUS_SCHEDULED, "예약됨"),
        (STATUS_COMPLETED, "완료"),
        (STATUS_CANCELLED, "취소"),
    ]

    TYPE_GENERAL = "예약"
    TYPE_EXAM = "검진"
    TYPE_MEETING = "회의"
    TYPE_INHOUSE = "내근"
    TYPE_OUTSIDE = "외근"
    TYPE_CHOICES = [
        (TYPE_GENERAL, "일반 예약"),
        (TYPE_EXAM, "검진"),
        (TYPE_MEETING, "회의"),
        (TYPE_INHOUSE, "내근"),
        (TYPE_OUTSIDE, "외근"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200, verbose_name="제목")
    type = models.CharField(max_length=30, choices=TYPE_CHOICES, default=TYPE_GENERAL, verbose_name="유형")
    start_time = models.DateTimeField(verbose_name="시작 일시")
    end_time = models.DateTimeField(null=True, blank=True, verbose_name="종료 일시")

    patient = models.ForeignKey(
        Patient,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="appointments",
        verbose_name="환자",
    )
    patient_identifier = models.CharField(max_length=50, blank=True, verbose_name="환자 ID")
    patient_name = models.CharField(max_length=100, blank=True, verbose_name="환자 이름")
    patient_gender = models.CharField(
        max_length=1,
        choices=Patient.GENDER_CHOICES,
        blank=True,
        verbose_name="환자 성별",
    )
    patient_age = models.PositiveIntegerField(null=True, blank=True, verbose_name="환자 나이")

    doctor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name="appointments",
        verbose_name="담당 의사",
    )
    doctor_code = models.CharField(max_length=20, blank=True, default="", verbose_name="의사 코드")
    doctor_username = models.CharField(max_length=150, verbose_name="의사 계정")
    doctor_name = models.CharField(max_length=150, blank=True, verbose_name="의사 이름")
    doctor_department = models.CharField(max_length=30, blank=True, verbose_name="진료과")

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_SCHEDULED, verbose_name="상태")
    memo = models.TextField(blank=True, verbose_name="메모")

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="created_appointments",
        verbose_name="등록자",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="등록일")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일")

    class Meta:
        verbose_name = "예약"
        verbose_name_plural = "예약들"
        ordering = ["-start_time", "-created_at"]

    def __str__(self):
        return f"{self.title} ({self.start_time:%Y-%m-%d %H:%M})"

    def save(self, *args, **kwargs):
        if not self.doctor_username:
            self.doctor_username = self.doctor.get_username()
        if not self.doctor_name:
            full_name = " ".join(filter(None, [self.doctor.last_name, self.doctor.first_name])).strip()
            self.doctor_name = full_name or self.doctor_username
        if not self.doctor_code:
            self.doctor_code = get_doctor_id(self.doctor.id) or ""
        if not self.doctor_department:
            self.doctor_department = get_department(self.doctor.id) or ""
        if self.patient and not self.patient_name:
            self.patient_name = self.patient.name
        if self.patient and not self.patient_identifier:
            self.patient_identifier = self.patient.patient_id
        if self.patient and not self.patient_gender:
            self.patient_gender = self.patient.gender or ""
        if self.patient and self.patient_age is None:
            self.patient_age = self.patient.age
        super().save(*args, **kwargs)

    @property
    def doctor_id(self) -> str:
        return self.doctor_code

    @doctor_id.setter
    def doctor_id(self, value: str) -> None:
        self.doctor_code = value

    @property
    def patient_id(self) -> str:
        return self.patient_identifier

    @patient_id.setter
    def patient_id(self, value: str) -> None:
        self.patient_identifier = value
