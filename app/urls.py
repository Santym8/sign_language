from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
     path('predict_hand_sign/', views.predict_hand_sign, name='predict_hand_sign'),
]

