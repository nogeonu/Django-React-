from rest_framework import viewsets, filters, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend
from .models import Patient, MedicalRecord
from .serializers import PatientSerializer, MedicalRecordSerializer, PatientUserSignupSerializer

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

class MedicalRecordViewSet(viewsets.ModelViewSet):
    queryset = MedicalRecord.objects.all()
    serializer_class = MedicalRecordSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['patient']
    search_fields = ['diagnosis', 'symptoms']
    ordering_fields = ['visit_date']
    ordering = ['-visit_date']


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
