from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    PatientViewSet,
    MedicalRecordViewSet,
    PatientSignupView,
    PatientLoginView,
    PatientProfileView,
    AppointmentViewSet,
)

router = DefaultRouter()
router.register(r'patients', PatientViewSet)
router.register(r'records', MedicalRecordViewSet)
router.register(r'appointments', AppointmentViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('signup/', PatientSignupView.as_view(), name='patient-signup'),
    path('login/', PatientLoginView.as_view(), name='patient-login'),
    path('profile/<str:account_id>/', PatientProfileView.as_view(), name='patient-profile'),
]
