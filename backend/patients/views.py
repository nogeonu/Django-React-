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
        except Exception as e:  # pragma: no cover
            import traceback
            print(f"[환자 회원가입 VIEW] 에러 발생: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            return Response(
                {
                    "detail": "환자 계정 생성 중 오류가 발생했습니다.",
                    "error": str(e),
                    "error_type": type(e).__name__
                },
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
    permission_classes = [permissions.AllowAny]  # 환자도 예약 가능하도록 변경
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['doctor', 'status', 'type']
    search_fields = ['title', 'patient_name', 'patient_id', 'doctor_username', 'doctor_name', 'memo']
    ordering_fields = ['start_time', 'created_at']
    ordering = ['start_time']  # 가까운 일정 순으로 정렬
    pagination_class = None  # 페이지네이션 비활성화 - 모든 예약 데이터 반환

    def get_queryset(self):
        queryset = super().get_queryset()
        
        # patient_id로 필터링 (patient_identifier 필드 사용)
        patient_id = self.request.query_params.get('patient_id')
        if patient_id:
            queryset = queryset.filter(patient_identifier=patient_id)
        
        return queryset

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

    def perform_create(self, serializer):
        print(f"[예약 등록] 요청 데이터: {self.request.data}")
        print(f"[예약 등록] 사용자: {self.request.user}")
        print(f"[예약 등록] 인증 여부: {self.request.user.is_authenticated}")
        if self.request.user.is_authenticated:
            serializer.save(created_by=self.request.user)
        else:
            serializer.save()

    def perform_update(self, serializer):
        serializer.save()
    
    @action(detail=False, methods=['get'])
    def my_appointments(self, request):
        """환자 ID로 예약 목록 조회"""
        patient_id = request.query_params.get('patient_id')
        if not patient_id:
            return Response(
                {"detail": "patient_id 파라미터가 필요합니다."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        appointments = self.queryset.filter(patient_identifier=patient_id).order_by('start_time')
        serializer = self.get_serializer(appointments, many=True)
        return Response(serializer.data)
