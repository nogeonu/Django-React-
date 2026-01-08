from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from . import views, orthanc_views, mammography_ai_views, mri_ai_views, yolo_detection_views, segmentation_views

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
    path('orthanc/patients/<str:patient_id>/delete/', orthanc_views.orthanc_delete_patient, name='orthanc-delete-patient'),
    
    # AI Segmentation API (MRI - 향후 모델 통합)
    path('orthanc/patients/<str:patient_id>/segmentation/', orthanc_views.orthanc_segmentation, name='orthanc-segmentation'),
    path('orthanc/patients/<str:patient_id>/segmentation/run/', orthanc_views.orthanc_run_segmentation, name='orthanc-run-segmentation'),
    
    # Mammography AI Detection API (YOLO11)
    path('mammography/instances/<str:instance_id>/detect/', mammography_ai_views.mammography_ai_detection, name='mammography-ai-detection'),
    path('mammography/ai/health/', mammography_ai_views.mammography_ai_health, name='mammography-ai-health'),
    
    # MRI AI Analysis API (pCR Prediction)
    path('patients/<str:patient_id>/analyze/', mri_ai_views.mri_ai_analysis, name='mri-ai-analysis'),
    path('ai/health/', mri_ai_views.mri_ai_health, name='mri-ai-health'),
    
    # YOLO 디텍션 API (FastAPI 서버 사용)
    path('yolo/instances/<str:instance_id>/detect/', yolo_detection_views.yolo_detection, name='yolo-detection'),
    path('yolo/health/', yolo_detection_views.yolo_health, name='yolo-health'),
    
    # MRI 세그멘테이션 API
    path('segmentation/instances/<str:instance_id>/segment/', segmentation_views.mri_segmentation, name='mri-segmentation'),
    path('segmentation/series/<str:series_id>/segment/', segmentation_views.segment_series, name='segment-series'),
    path('segmentation/health/', segmentation_views.segmentation_health, name='segmentation-health'),
]

