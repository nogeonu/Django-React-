from rest_framework import viewsets, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from .models import Patient, MedicalRecord
from .serializers import PatientSerializer, MedicalRecordSerializer

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
