# learning/admin.py

from django.contrib import admin
from .models import (
    Module, Lesson, Quiz, Question, Answer, Mastery, 
    UserLessonProgress, Badge, Flashcard, UserFlashcard
)

# This class will allow you to add Answers directly on the Question page
class AnswerInline(admin.TabularInline):
    model = Answer
    extra = 4  # Provides 4 blank answer slots by default

# This class customizes how the Question model is displayed
class QuestionAdmin(admin.ModelAdmin):
    inlines = [AnswerInline] # Adds the Answer inline to the Question page


# This will allow you to add Questions directly on the Quiz page
class QuestionInline(admin.TabularInline):
    model = Question
    extra = 1

class QuizAdmin(admin.ModelAdmin):
    inlines = [QuestionInline]


# learning/admin.py
from .models import  QuizAttempt, UserAnswer # Add to import

admin.site.register(QuizAttempt)
admin.site.register(UserAnswer)

# Register your models with the custom admin classes
admin.site.register(Question, QuestionAdmin)
admin.site.register(Quiz, QuizAdmin)

# Register the rest of the models normally
admin.site.register(Module)
admin.site.register(Lesson)
admin.site.register(Mastery)
admin.site.register(UserLessonProgress)
admin.site.register(Badge)
admin.site.register(Flashcard)
admin.site.register(UserFlashcard)

# learning/admin.py
from .models import  StudyPlan # Add StudyPlan to the import

# ...
admin.site.register(StudyPlan) # Add this line

# learning/admin.py
from .models import UserInteraction # Add to import

admin.site.register(UserInteraction) # Add this line

# learning/admin.py
from .models import  ScheduledDay, ScheduledTask,KnowledgeGap,Notification,RevisionCard# Add to import

admin.site.register(ScheduledDay)
admin.site.register(ScheduledTask)
admin.site.register(KnowledgeGap)
admin.site.register(Notification)
admin.site.register(RevisionCard)
