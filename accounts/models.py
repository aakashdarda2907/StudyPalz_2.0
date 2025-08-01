# accounts/models.py
from django.db import models
from django.contrib.auth.models import User
from learning.models import Badge

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    badges = models.ManyToManyField(Badge, blank=True)

    LEARNING_STYLE_CHOICES = [
        ('VISUAL', 'Visual'),
        ('AUDITORY', 'Auditory'),
        ('KINESTHETIC', 'Kinesthetic/Code-First'),
        ('READING', 'Reading/Writing'),
    ]
    learning_style = models.CharField(max_length=20, choices=LEARNING_STYLE_CHOICES, default='VISUAL')
    interests = models.TextField(blank=True, help_text="Comma-separated interests, e.g., 'finance, sports'")

    def __str__(self):
        return f"{self.user.username}'s Profile"