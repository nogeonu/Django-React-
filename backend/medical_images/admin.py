from django.contrib import admin
from .models import MedicalImage

@admin.register(MedicalImage)
class MedicalImageAdmin(admin.ModelAdmin):
    list_display = ['patient', 'image_type', 'taken_date', 'created_at']
    list_filter = ['image_type', 'taken_date', 'created_at']
    search_fields = ['patient__name', 'description', 'doctor_notes']
    ordering = ['-taken_date']
