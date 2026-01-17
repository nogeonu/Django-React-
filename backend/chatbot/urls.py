from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),
    path('skin/analyze/', views.skin_analyze, name='skin_analyze'),
]
