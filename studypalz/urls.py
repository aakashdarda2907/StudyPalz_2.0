# studypalz/urls.py

from django.contrib import admin
from django.urls import path, include # <-- Make sure to import 'include'

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # Delegate URLs starting with 'accounts/' to the accounts app
    path('accounts/', include('accounts.urls')), 
    
    # For now, delegate the root URL and others to the learning app
    path('', include('learning.urls')), 
]