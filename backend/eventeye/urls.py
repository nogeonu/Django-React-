"""
URL configuration for eventeye project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import JsonResponse

def api_root(request):
    return JsonResponse({
        'message': 'EventEye API',
        'version': '1.0.0',
        'endpoints': {
            'patients': '/api/patients/',
            'medical_images': '/api/medical-images/',
            'dashboard': '/api/dashboard/',
            'lung_cancer': '/api/lung_cancer/',
            'admin': '/admin/'
        }
    })

urlpatterns = [
    path('', api_root, name='root'),
    path('admin/', admin.site.urls),
    path('api/', api_root, name='api-root'),
    path('api/patients/', include('patients.urls')),
    path('api/medical-images/', include('medical_images.urls')),
    path('api/dashboard/', include('dashboard.urls')),
    path('api/lung_cancer/', include('lung_cancer.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
