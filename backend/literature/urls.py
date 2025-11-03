from django.urls import path
from .views import LitSearchView, LitOpenAccessView, NewsFeedView

urlpatterns = [
    path("search", LitSearchView.as_view()),
    path("openaccess", LitOpenAccessView.as_view()),
    path("news", NewsFeedView.as_view()),
]
