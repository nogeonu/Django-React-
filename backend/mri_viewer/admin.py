from django.contrib import admin
from .models import MRIStudy


@admin.register(MRIStudy)
class MRIStudyAdmin(admin.ModelAdmin):
    list_display = ['patient_id', 'study_date', 'scanner_manufacturer', 'age', 'created_at']
    list_filter = ['scanner_manufacturer', 'tumor_subtype', 'study_date']
    search_fields = ['patient_id']
    ordering = ['-created_at']

