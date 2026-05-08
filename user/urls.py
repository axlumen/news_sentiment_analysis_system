from django.urls import path
from django.views.generic import RedirectView
from user import views

app_name = 'user'
urlpatterns = [
    path('', RedirectView.as_view(url='login/', permanent=False), name='index'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('profile/edit/', views.edit_profile_view, name='edit_profile'),
    path('profile/avatar/', views.upload_avatar_view, name='upload_avatar'),
    path('profile/password/', views.change_password_view, name='change_password'),
]