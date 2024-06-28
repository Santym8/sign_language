from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('save_photo/', views.save_photo, name='save_photo'),
]

