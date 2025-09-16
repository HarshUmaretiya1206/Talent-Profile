from django.urls import path
from . import views


urlpatterns = [
    path('health', views.health, name='health'),
    path('dataset', views.dataset_info, name='dataset-info'),
    path('recommend', views.recommend, name='recommend'),
    path('talent/<int:idx>', views.talent_detail, name='talent-detail'),
]


