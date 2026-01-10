from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from . import views, orthanc_views, segmentation_views, mammography_views

urlpatterns = [
    # 기존 MRI Viewer API
    path('patients/', views.get_patient_list, name='mri-patient-list'),
    path('patients/<str:patient_id>/', views.get_patient_basic_info, name='mri-patient-basic-info'),
    path('patients/<str:patient_id>/detail/', views.get_patient_info, name='mri-patient-info'),
    path('patients/<str:patient_id>/slice/', views.get_mri_slice, name='mri-slice'),
    path('patients/<str:patient_id>/volume/', views.get_volume_info, name='mri-volume-info'),
    
    # Orthanc PACS API
    path('orthanc/system/', orthanc_views.orthanc_system_info, name='orthanc-system'),
    path('orthanc/debug/patients/', orthanc_views.orthanc_debug_patients, name='orthanc-debug-patients'),
    path('orthanc/patients/', orthanc_views.orthanc_patients, name='orthanc-patients'),
    path('orthanc/patients/<str:patient_id>/', orthanc_views.orthanc_patient_detail, name='orthanc-patient-detail'),
    path('orthanc/instances/<str:instance_id>/preview/', orthanc_views.orthanc_instance_preview, name='orthanc-instance-preview'),
    path('orthanc/instances/<str:instance_id>/file', orthanc_views.orthanc_instance_file, name='orthanc-instance-file'),
    path('orthanc/upload/', csrf_exempt(orthanc_views.orthanc_upload_dicom), name='orthanc-upload'),
    path('orthanc/upload-folder/', csrf_exempt(orthanc_views.orthanc_upload_dicom_folder), name='orthanc-upload-folder'),
    path('orthanc/patients/<str:patient_id>/delete/', orthanc_views.orthanc_delete_patient, name='orthanc-delete-patient'),
    
    # AI Segmentation API (MRI - 향후 모델 통합)
    path('orthanc/patients/<str:patient_id>/segmentation/', orthanc_views.orthanc_segmentation, name='orthanc-segmentation'),
    path('orthanc/patients/<str:patient_id>/segmentation/run/', orthanc_views.orthanc_run_segmentation, name='orthanc-run-segmentation'),
    
    # MRI 세그멘테이션 API
    path('segmentation/instances/<str:instance_id>/segment/', segmentation_views.mri_segmentation, name='mri-segmentation'),
    path('segmentation/series/<str:series_id>/segment/', segmentation_views.segment_series, name='segment-series'),
    path('segmentation/instances/<str:seg_instance_id>/frames/', segmentation_views.get_segmentation_frames, name='get-segmentation-frames'),
    path('segmentation/health/', segmentation_views.segmentation_health, name='segmentation-health'),
    
    # 맘모그래피 AI 분석 API
    path('mammography/analyze/', mammography_views.analyze_mammography, name='analyze-mammography'),
    path('mammography/health/', mammography_views.mammography_health, name='mammography-health'),
]

