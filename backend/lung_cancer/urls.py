from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'patients', views.PatientViewSet)
router.register(r'records', views.LungRecordViewSet)
router.register(r'results', views.LungResultViewSet)
router.register(r'medical-records', views.MedicalRecordViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('visualization/', views.visualization_data, name='visualization_data'),
]
