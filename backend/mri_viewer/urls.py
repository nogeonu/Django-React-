from django.urls import path
from . import views, orthanc_views

urlpatterns = [
    # 기존 MRI Viewer API
    path('patients/', views.get_patient_list, name='mri-patient-list'),
    path('patients/<str:patient_id>/', views.get_patient_info, name='mri-patient-info'),
    path('patients/<str:patient_id>/slice/', views.get_mri_slice, name='mri-slice'),
    path('patients/<str:patient_id>/volume/', views.get_volume_info, name='mri-volume-info'),
    
    # Orthanc PACS API
    path('orthanc/system/', orthanc_views.orthanc_system_info, name='orthanc-system'),
    path('orthanc/patients/', orthanc_views.orthanc_patients, name='orthanc-patients'),
    path('orthanc/patients/<str:patient_id>/', orthanc_views.orthanc_patient_detail, name='orthanc-patient-detail'),
    path('orthanc/instances/<str:instance_id>/preview/', orthanc_views.orthanc_instance_preview, name='orthanc-instance-preview'),
    path('orthanc/upload/', orthanc_views.orthanc_upload_dicom, name='orthanc-upload'),
    path('orthanc/patients/<str:patient_id>/delete/', orthanc_views.orthanc_delete_patient, name='orthanc-delete-patient'),
]

