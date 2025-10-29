from rest_framework import viewsets, filters
from django_filters.rest_framework import DjangoFilterBackend
from .models import MedicalImage
from .serializers import MedicalImageSerializer

class MedicalImageViewSet(viewsets.ModelViewSet):
    queryset = MedicalImage.objects.all()
    serializer_class = MedicalImageSerializer
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['image_type', 'patient']
    search_fields = ['description', 'doctor_notes', 'patient__name']
    ordering_fields = ['taken_date', 'created_at']
    ordering = ['-taken_date']
