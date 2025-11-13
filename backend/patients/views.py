from rest_framework import viewsets, filters, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend
from django.http import Http404
from .models import Patient, MedicalRecord, PatientUser, Appointment
from .serializers import (
    PatientSerializer,
    MedicalRecordSerializer,
    PatientUserSignupSerializer,
    PatientProfileSerializer,
    AppointmentSerializer,
)

class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['gender']
    search_fields = ['name', 'patient_id', 'phone']
    ordering_fields = ['created_at', 'name']
    ordering = ['-created_at']
    
    @action(detail=True, methods=['get'])
    def medical_records(self, request, pk=None):
        patient = self.get_object()
        records = patient.medical_records.all()
        serializer = MedicalRecordSerializer(records, many=True)
        return Response(serializer.data)


class PatientSignupView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes: list = []

    def post(self, request):
        serializer = PatientUserSignupSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        try:
            user = serializer.save()
        except Exception:  # pragma: no cover
            return Response(
                {"detail": "환자 계정 생성 중 오류가 발생했습니다."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {
                "message": "환자 계정이 생성되었습니다.",
                "account_id": user.account_id,
                "email": user.email,
                "patient_id": user.patient_id,
            },
            status=status.HTTP_201_CREATED,
        )


class PatientLoginView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes: list = []

    def post(self, request):
        account_id = request.data.get("account_id", "").strip()
        password = request.data.get("password", "")

        if not account_id or not password:
            return Response(
                {"detail": "계정 ID와 비밀번호를 모두 입력해주세요."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            user = PatientUser.objects.get(account_id=account_id)
        except PatientUser.DoesNotExist:
            return Response(
                {"detail": "계정 ID 또는 비밀번호가 올바르지 않습니다."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not user.check_password(password):
            return Response(
                {"detail": "계정 ID 또는 비밀번호가 올바르지 않습니다."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(
            {
                "message": "로그인에 성공했습니다.",
                "account_id": user.account_id,
                "patient_id": user.patient_id,
                "name": user.name,
                "email": user.email,
                "phone": user.phone,
            },
            status=status.HTTP_200_OK,
        )


class PatientProfileView(APIView):
    permission_classes = [permissions.AllowAny]
    authentication_classes: list = []

    def get_patient(self, account_id: str):
        try:
            patient_user = PatientUser.objects.get(account_id=account_id)
        except PatientUser.DoesNotExist:
            raise Http404

        patient = getattr(patient_user, "patient_profile", None)
        if not patient:
            raise Http404
        return patient, patient_user

    def get(self, request, account_id: str):
        patient, _ = self.get_patient(account_id)
        serializer = PatientProfileSerializer(patient)
        return Response(serializer.data)

    def put(self, request, account_id: str):
        patient, patient_user = self.get_patient(account_id)
        data = request.data.copy()

        for field in ["birth_date", "gender", "blood_type"]:
            if data.get(field) == "":
                data[field] = None

        serializer = PatientProfileSerializer(patient, data=data, partial=True)
        serializer.is_valid(raise_exception=True)
        updated_patient = serializer.save()

        has_changes = False
        if "name" in serializer.validated_data:
            patient_user.name = updated_patient.name
            has_changes = True
        if "phone" in serializer.validated_data:
            patient_user.phone = updated_patient.phone
            has_changes = True
        if has_changes:
            patient_user.save(update_fields=["name", "phone"])

        return Response(PatientProfileSerializer(updated_patient).data)


class MedicalRecordViewSet(viewsets.ModelViewSet):
    queryset = MedicalRecord.objects.all()
    serializer_class = MedicalRecordSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['patient']
    search_fields = ['diagnosis', 'symptoms']
    ordering_fields = ['visit_date']
    ordering = ['-visit_date']


class AppointmentViewSet(viewsets.ModelViewSet):
    queryset = Appointment.objects.select_related('patient', 'doctor', 'created_by').all()
    serializer_class = AppointmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['doctor', 'status', 'type', 'patient_id']
    search_fields = ['title', 'patient_name', 'patient_id', 'doctor_username', 'doctor_name', 'memo']
    ordering_fields = ['start_time', 'created_at']
    ordering = ['-start_time']

    def get_queryset(self):
        queryset = super().get_queryset()
        # 의료진은 본인이 담당한 예약만 필터링할 수도 있음. 기본은 전체 조회.
        return queryset

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

    def perform_create(self, serializer):
        print(f"[예약 등록] 요청 데이터: {self.request.data}")
        print(f"[예약 등록] 사용자: {self.request.user}")
        print(f"[예약 등록] 인증 여부: {self.request.user.is_authenticated}")
        serializer.save(created_by=self.request.user)

    def perform_update(self, serializer):
        serializer.save()
