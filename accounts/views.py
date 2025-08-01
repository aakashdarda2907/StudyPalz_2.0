# accounts/views.py

from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import UserProfile

def register_view(request):
    if request.method == 'POST':
        # Get form data
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password_confirm = request.POST.get('password_confirm')

        # Validate form data
        if password != password_confirm:
            messages.error(request, 'Passwords do not match.')
            return redirect('register')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username is already taken.')
            return redirect('register')

       # Create new user
        user = User.objects.create_user(username=username, email=email, password=password)
        
        # 2. CREATE THE PROFILE IMMEDIATELY AFTER CREATING THE USER
        # While this works, the Signal method is recommended.
        UserProfile.objects.create(user=user)

        
        # Log the user in and redirect to home page
        login(request, user)
        messages.success(request, f'Welcome, {username}! Your account has been created.')
        return redirect('dashboard') # We will create the 'home' URL name in a later task

    return render(request, 'accounts/register.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.info(request, f'Welcome back, {username}!')
            return redirect('dashboard') # We will create the 'home' URL name later
        else:
            messages.error(request, 'Invalid username or password.')
            return redirect('login')

    return render(request, 'accounts/login.html')


def logout_view(request):
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')

