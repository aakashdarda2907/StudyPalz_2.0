# In learning/management/commands/run_daily_maintenance.py

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from learning.ai_utils import perform_daily_schedule_maintenance
from django.utils import timezone

class Command(BaseCommand):
    help = 'Runs the daily schedule maintenance for all active users.'

    def handle(self, *args, **options):
        self.stdout.write(f"Starting daily schedule maintenance at {timezone.now()}...")
        
        # Get all users you want to run this for (e.g., all users)
        users = User.objects.all()
        
        for user in users:
            self.stdout.write(f"  -> Processing schedule for user: {user.username}")
            insights = perform_daily_schedule_maintenance(user)
            
            if insights:
                for insight in insights:
                    # You could save these insights to a notification model later
                    self.stdout.write(self.style.SUCCESS(f"    - Insight: {insight}"))
            else:
                self.stdout.write("    - No schedule changes needed.")
                
        self.stdout.write(self.style.SUCCESS("Daily maintenance complete."))