from rest_framework import viewsets, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from .models import Patient, MedicalRecord, Appointment
from .serializers import PatientSerializer, MedicalRecordSerializer, AppointmentSerializer

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

class AppointmentViewSet(viewsets.ModelViewSet):
    queryset = Appointment.objects.all()
    serializer_class = AppointmentSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['patient', 'status', 'appointment_type']
    search_fields = ['title', 'description', 'patient__name', 'patient__patient_id']
    ordering_fields = ['appointment_date', 'created_at']
    ordering = ['-appointment_date']
    
    def perform_create(self, serializer):
        # 현재 로그인한 사용자를 created_by로 설정
        serializer.save(created_by=self.request.user if self.request.user.is_authenticated else None)
    
    @action(detail=False, methods=['get'])
    def by_patient(self, request):
        """환자 ID로 예약 조회"""
        patient_id = request.query_params.get('patient_id', None)
        if patient_id:
            try:
                patient = Patient.objects.get(patient_id=patient_id)
                appointments = Appointment.objects.filter(patient=patient)
                serializer = self.get_serializer(appointments, many=True)
                return Response(serializer.data)
            except Patient.DoesNotExist:
                return Response({'error': '환자를 찾을 수 없습니다.'}, status=404)
        return Response({'error': 'patient_id 파라미터가 필요합니다.'}, status=400)
