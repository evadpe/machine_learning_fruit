from django.urls import path
from . import views

app_name = 'fruits_identification'

urlpatterns = [
    path('', views.accueil, name='accueil'),
    path('identifier/', views.identifier_fruit, name='identifier'),
]
