from django.urls import path
from . import views

urlpatterns = [
    path('classify-shot/', views.classify_shot, name='classify_shot'),
    path('health/', views.health_check, name='health_check'),
] 