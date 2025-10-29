from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'patients', views.PatientViewSet)
router.register(r'records', views.LungRecordViewSet)
router.register(r'results', views.LungResultViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/visualization/', views.visualization_data, name='visualization_data'),
]
