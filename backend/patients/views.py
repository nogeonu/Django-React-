from rest_framework import viewsets, filters, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend
from django.http import Http404
import logging
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
    # queryset을 제거하고 get_queryset만 사용
    # queryset = Appointment.objects.select_related('patient', 'doctor', 'created_by').all()
    serializer_class = AppointmentSerializer
    permission_classes = [permissions.AllowAny]  # 환자도 예약 가능하도록 변경
    # filter_backends를 제거하여 get_queryset의 필터링만 사용
    # filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    # filterset_fields = ['doctor', 'status', 'type', 'doctor_code']
    search_fields = ['title', 'patient_name', 'patient_id', 'doctor_username', 'doctor_name', 'memo']
    ordering_fields = ['start_time', 'created_at']
    ordering = ['start_time']  # 가까운 일정 순으로 정렬
    pagination_class = None  # 페이지네이션 비활성화 - 모든 예약 데이터 반환

    def get_queryset(self):
        # 기본 쿼리셋 생성 (queryset 클래스 속성이 없으므로 직접 생성)
        queryset = Appointment.objects.select_related('patient', 'doctor', 'created_by').all()
        
        # 기본적으로 취소되지 않은 예약만 조회
        # status 파라미터가 명시적으로 전달된 경우에만 해당 상태의 예약 조회
        status = self.request.query_params.get('status')
        if status is None:
            # status 파라미터가 없으면 취소된 예약 제외
            queryset = queryset.exclude(status='cancelled')
        
        # 로그인한 사용자가 의사인 경우, 자신의 진료과 일정만 조회
        # 원무과는 모든 진료과 일정 조회 가능
        if self.request.user.is_authenticated:
            import logging
            logger = logging.getLogger(__name__)
            from eventeye.doctor_utils import get_department
            user_department = get_department(self.request.user.id)
            
            logger.info(f"[예약 필터링] 사용자: {self.request.user.username} (ID: {self.request.user.id}), 부서: {user_department}")
            
            # 원무과가 아니고 의료진인 경우, 자신의 진료과 일정만 조회
            if user_department and user_department != "원무과":
                # doctor_department 필드로 직접 필터링 (가장 효율적)
                from django.db.models import Q
                
                # 방법 1: doctor_department가 정확히 일치하는 예약
                matching_by_dept = queryset.filter(doctor_department=user_department)
                matching_ids = set(matching_by_dept.values_list('id', flat=True))
                logger.info(f"[예약 필터링] doctor_department='{user_department}'인 예약: {len(matching_ids)}개")
                
                # 방법 2: doctor_department가 비어있거나 None인 예약 중에서 doctor의 부서 확인
                empty_dept_queryset = queryset.filter(
                    Q(doctor_department='') | Q(doctor_department__isnull=True),
                    doctor__isnull=False
                )
                logger.info(f"[예약 필터링] doctor_department가 비어있는 예약: {empty_dept_queryset.count()}개")
                
                # doctor의 부서가 일치하는 예약 찾기 및 doctor_department 업데이트
                for appointment in empty_dept_queryset:
                    if appointment.doctor:
                        doctor_dept = get_department(appointment.doctor.id)
                        logger.info(f"[예약 필터링] 예약 ID {appointment.id}: doctor_department='{appointment.doctor_department}', doctor 부서='{doctor_dept}', user_department='{user_department}'")
                        if doctor_dept == user_department:
                            matching_ids.add(appointment.id)
                            # doctor_department 업데이트 (다음 조회 시 효율적)
                            if not appointment.doctor_department:
                                appointment.doctor_department = user_department
                                appointment.save(update_fields=['doctor_department'])
                                logger.info(f"[예약 필터링] 예약 ID {appointment.id}의 doctor_department 업데이트: '{user_department}'")
                
                # 필터링된 ID로 queryset 재구성
                logger.info(f"[예약 필터링] 최종 필터링된 예약 ID 수: {len(matching_ids)}개")
                if matching_ids:
                    queryset = queryset.filter(id__in=matching_ids)
                else:
                    # 매칭되는 예약이 없으면 빈 쿼리셋 반환
                    queryset = queryset.none()
                    logger.info(f"[예약 필터링] 매칭되는 예약이 없어 빈 쿼리셋 반환")
                
                # 디버깅: 필터링 후 실제 반환되는 예약 확인
                final_count = queryset.count()
                logger.info(f"[예약 필터링] 최종 queryset.count(): {final_count}개")
                if final_count > 0:
                    sample_appointments = queryset[:5]
                    for app in sample_appointments:
                        logger.info(f"[예약 필터링] 반환 예약 샘플 - ID: {app.id}, doctor_department: '{app.doctor_department}', title: '{app.title}'")
                else:
                    logger.warning(f"[예약 필터링] 필터링 후 예약이 없습니다. user_department='{user_department}'")
        
        # patient_id로 필터링 (patient_identifier 필드 사용)
        patient_id = self.request.query_params.get('patient_id')
        if patient_id:
            queryset = queryset.filter(patient_identifier=patient_id)
        
        # doctor_code로 필터링
        doctor_code = self.request.query_params.get('doctor_code')
        if doctor_code:
            queryset = queryset.filter(doctor_code=doctor_code)
        
        return queryset
    
    def list(self, request, *args, **kwargs):
        """목록 조회 시 부서별 필터링 강제 적용"""
        print(f"[list 메서드 시작] 사용자: {request.user.username}")
        
        # get_queryset 대신 직접 필터링
        queryset = Appointment.objects.select_related('patient', 'doctor', 'created_by').all()
        print(f"[list 메서드] 전체 예약: {queryset.count()}개")
        
        # 취소된 예약 제외
        queryset = queryset.exclude(status='cancelled')
        print(f"[list 메서드] 취소 제외 후: {queryset.count()}개")
        
        # 부서별 필터링
        if request.user.is_authenticated:
            from eventeye.doctor_utils import get_department
            from django.db.models import Q
            
            user_department = get_department(request.user.id)
            print(f"[list 메서드] 사용자 부서: {user_department}")
            
            if user_department and user_department != "원무과":
                # 직접 필터링
                before_count = queryset.count()
                queryset = queryset.filter(doctor_department=user_department)
                after_count = queryset.count()
                print(f"[list 메서드] 부서 필터링: {before_count}개 -> {after_count}개")
        
        print(f"[list 메서드] 최종 반환: {queryset.count()}개")
        
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

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
