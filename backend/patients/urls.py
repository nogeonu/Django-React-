from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PatientViewSet, MedicalRecordViewSet, PatientSignupView, PatientLoginView

router = DefaultRouter()
router.register(r'patients', PatientViewSet)
router.register(r'records', MedicalRecordViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('signup/', PatientSignupView.as_view(), name='patient-signup'),
    path('login/', PatientLoginView.as_view(), name='patient-login'),
]
