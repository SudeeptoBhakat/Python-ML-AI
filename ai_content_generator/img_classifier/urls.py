from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('classify-image/', views.imageupload, name='classify_image'),
]