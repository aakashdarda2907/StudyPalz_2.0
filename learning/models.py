# learning/models.py
from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# ==============================================================================
# --- Core Plan & Content Models ---
# ==============================================================================

class StudyPlan(models.Model):
    """Represents a user-generated study curriculum."""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    exam_date = models.DateField(null=True, blank=True)
    subject = models.CharField(max_length=100, blank=True)


    def __str__(self):
        return f"{self.user.username}'s plan: {self.title}"

class Module(models.Model):
    """A logical grouping of lessons within a StudyPlan, like a chapter."""
    study_plan = models.ForeignKey(StudyPlan, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    order = models.IntegerField()

    def __str__(self):
        return self.title

class Lesson(models.Model):
    """A single, teachable unit of content within a Module."""
    module = models.ForeignKey(Module, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    content = models.TextField(blank=True, help_text="AI-generated revision notes are stored here.")
    concept_key = models.CharField(max_length=100, unique=True)
    order = models.IntegerField()
    video_url = models.URLField(blank=True, null=True)
    revision_notes = models.TextField(blank=True, null=True)
    analogy = models.TextField(blank=True, null=True)
    code_example = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.module.title} - {self.title}"

# ==============================================================================
# --- Quiz & Assessment Models ---
# ==============================================================================
# learning/models.py

# (Keep all your other models like StudyPlan, Module, Lesson, etc.)

# ==============================================================================
# --- Unified Quiz & Assessment Models ---
# ==============================================================================
# In learning/models.py, replace the existing Quiz model with this one
# In learning/models.py, replace the existing Quiz model with this one

class Quiz(models.Model):
    """ A flexible quiz that can be a standard lesson quiz or an automated, personalized one. """
    
    # --- Constants for quiz types ---
    LESSON_QUIZ = 'lesson'
    DAILY_QUIZ = 'daily'
    WEEKLY_QUIZ = 'weekly'
    SURPRISE_QUIZ = 'surprise'
    EXAM_POWER_PACK = 'exam_pack'

    QUIZ_TYPE_CHOICES = [
        (LESSON_QUIZ, 'Lesson Quiz'),
        (DAILY_QUIZ, 'Daily Review Quiz'),
        (WEEKLY_QUIZ, 'Weekly Progress Quiz'),
        (SURPRISE_QUIZ, 'Surprise Quiz'),
        (EXAM_POWER_PACK, 'Pre-Exam Power Pack'),
    ]
    
    # --- Fields ---
    # FIX: Made user nullable to handle migration with existing data
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, help_text="The user this quiz is for.")
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, null=True, blank=True, help_text="The lesson this quiz is based on, if any.")
    title = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    
    quiz_type = models.CharField(
        max_length=20,
        choices=QUIZ_TYPE_CHOICES,
        default=LESSON_QUIZ
    )
    ai_insight = models.TextField(
        blank=True, 
        null=True, 
        help_text="Explanation from the AI on why this quiz was generated."
    )

    def __str__(self):
        # A safer string representation in case user is None
        user_str = self.user.username if self.user else "No User"
        return f"[{self.get_quiz_type_display()}] {self.title} for {user_str}"

# In learning/models.py

class Question(models.Model):
    """ A single question within a Quiz. """
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='questions')
    # --- ADD THIS LINE ---
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, null=True, blank=True, help_text="The specific lesson this question is about.")
    text = models.TextField()
    category = models.CharField(
        max_length=50,
        choices=[('CONCEPTUAL', 'Conceptual'), ('CODE', 'Code'), ('TERMINOLOGY', 'Terminology')],
        default='CONCEPTUAL')
    question_type = models.CharField(
        max_length=20, 
        choices=[('multiple-choice', 'Multiple-Choice'), ('open-ended', 'Open-Ended')], 
        default='multiple-choice'
    )
    grading_rubric = models.TextField(blank=True, null=True, help_text="Rubric for grading open-ended questions.")

    def __str__(self):
        return self.text

class Answer(models.Model):
    """ A single answer choice for a Question. """
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='answers')
    text = models.CharField(max_length=500)
    is_correct = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.text} ({'Correct' if self.is_correct else 'Incorrect'})"

class QuizAttempt(models.Model):
    """ Stores the result of a single user attempt at a quiz. """
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    score = models.IntegerField()
    total_questions = models.IntegerField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def get_score_percent(self):
        return int((self.score / self.total_questions) * 100) if self.total_questions > 0 else 0

class UserAnswer(models.Model):
    """ Stores the specific answer a user selected for a question in an attempt. """
    quiz_attempt = models.ForeignKey(QuizAttempt, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    selected_answer = models.ForeignKey(Answer, on_delete=models.CASCADE, null=True, blank=True)
    
    is_correct = models.BooleanField(null=True, blank=True)

    open_ended_response = models.TextField(blank=True, null=True)
    ai_feedback = models.TextField(blank=True, null=True)
    ai_score = models.FloatField(null=True, blank=True, help_text="Score from 0.0 to 1.0 for open-ended answers")

# ==============================================================================
# --- User Progress & Analytics Models ---
# ==============================================================================
# In learning/models.py, replace the existing Mastery model with this one

# In learning/models.py, replace the existing Mastery model with this one

# In learning/models.py, replace the existing Mastery model

class Mastery(models.Model):
    """
    Stores the AI-calculated mastery and forgetting curve parameters 
    for a user on a specific lesson.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='mastery_levels')
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, null=True, blank=True)
    mastery_score = models.FloatField(default=0.0, help_text="Current mastery score from 0.0 to 1.0, updated after every quiz.")
    last_tested_date = models.DateTimeField(null=True, blank=True, help_text="The last time the user was quizzed on this lesson.")
    
    # --- NEW: Fields for Forgetting Curve Modeling ---
    memory_strength = models.FloatField(default=0.0, help_text="A calculated score representing the durability of this memory.")
    next_review_date = models.DateField(null=True, blank=True, help_text="The AI-predicted optimal date for the next review.")

    class Meta:
        unique_together = ('user', 'lesson')
        verbose_name_plural = "Masteries"

    def __str__(self):
        if self.lesson:
            return f"{self.user.username}'s mastery on {self.lesson.title}: {self.mastery_score:.2f}"
        return f"{self.user.username}'s mastery on an unlinked lesson: {self.mastery_score:.2f}"


    class Meta:
        # Ensures a user has only one mastery record per lesson
        unique_together = ('user', 'lesson')
        verbose_name_plural = "Masteries"

    def __str__(self):
        if self.lesson:
            return f"{self.user.username}'s mastery on {self.lesson.title}: {self.mastery_score:.2f}"
        return f"{self.user.username}'s mastery on an unlinked concept: {self.mastery_score:.2f}"

class UserLessonProgress(models.Model):
    """Tracks a user's completion of a lesson and their last quiz score."""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE)
    completed_at = models.DateTimeField(auto_now_add=True)
    quiz_score = models.FloatField(null=True, blank=True)

    class Meta:
        unique_together = ('user', 'lesson')

    def __str__(self):
        return f"{self.user.username} - {self.lesson.title}"

class UserInteraction(models.Model):
    """Logs a user's interaction with a piece of content for ML analysis."""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE)
    interaction_type = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)
    feedback = models.IntegerField(null=True, blank=True, choices=[(1, 'Helpful'), (-1, 'Not Helpful')])

    def __str__(self):
        return f"{self.user.username} - {self.interaction_type} on {self.lesson.title}"

# ==============================================================================
# --- Scheduler & Gamification Models ---
# ==============================================================================

class ScheduledDay(models.Model):
    """Represents a single day in a user's generated study schedule."""
    study_plan = models.ForeignKey(StudyPlan, on_delete=models.CASCADE)
    date = models.DateField()

    class Meta:
        unique_together = ('study_plan', 'date')

    def __str__(self):
        return f"Schedule for {self.study_plan.title} on {self.date}"

class ScheduledTask(models.Model):
    """Represents a single task on a scheduled day."""
    scheduled_day = models.ForeignKey(ScheduledDay, on_delete=models.CASCADE)
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, null=True, blank=True)
    task_description = models.CharField(max_length=500)
    completed = models.BooleanField(default=False)
    task_type = models.CharField(max_length=10, choices=[('STUDY', 'Study'), ('QUIZ', 'Quiz'), ('REVIEW', 'Review')], default='STUDY')
    triggering_attempt = models.ForeignKey(QuizAttempt, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"{self.task_type}: {self.task_description}"

class Badge(models.Model):
    """Represents an achievement or badge a user can earn."""
    name = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return self.name

# ==============================================================================
# --- Spaced Repetition Models ---
# ==============================================================================

# In learning/models.py, in the SPACED REPETITION MODELS section

class FlashcardSet(models.Model):
    """Represents a user-created deck or set of flashcards."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='flashcard_sets')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"'{self.title}' by {self.user.username}"

# In learning/models.py, replace the old Flashcard model with this one

class Flashcard(models.Model):
    """
    A single flashcard. It can be linked to a lesson (AI-generated)
    or a user-created set.
    """
    # Link to a lesson is now optional to allow for user-created cards
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, related_name='flashcards', null=True, blank=True)
    
    # NEW: Link to a user-created deck (optional)
    flashcard_set = models.ForeignKey(FlashcardSet, on_delete=models.CASCADE, related_name='flashcards', null=True, blank=True)
    
    # NEW: Track who created the card (null for AI, user for user-created)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    question = models.TextField()
    answer = models.TextField()

    def __str__(self):
        if self.lesson:
            return f"AI Flashcard for {self.lesson.title}"
        elif self.flashcard_set:
            return f"User card for set '{self.flashcard_set.title}'"
        return f"Standalone Flashcard ({self.id})"

class UserFlashcard(models.Model):
    """Tracks a specific user's progress on a specific flashcard for SRS."""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    flashcard = models.ForeignKey(Flashcard, on_delete=models.CASCADE)
    next_review_date = models.DateField(default=timezone.now)
    last_interval_days = models.IntegerField(default=1)

    class Meta:
        unique_together = ('user', 'flashcard')
    
    def __str__(self):
        return f"{self.user.username}'s card: {self.flashcard.question[:50]}"
    
# learning/models.py

# New KnowledgeGap Model
class KnowledgeGap(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    identified_date = models.DateTimeField(auto_now_add=True)
    gap_details = models.JSONField()  # To store related topics, weaknesses, etc.
    ai_summary = models.TextField()

    def __str__(self):
        return f"Knowledge Gap for {self.user.username} - {self.identified_date.strftime('%Y-%m-%d')}"
    

# In learning/models.py, add this new model

class Notification(models.Model):
    """Stores a user-facing message, typically generated by an AI process."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    
    def __str__(self):
        return f"Notification for {self.user.username}: {self.message[:50]}"
    
# In learning/models.py, add this new model at the end

class PowerPack(models.Model):
    """Stores a user's generated Pre-Exam Power Pack."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='power_packs')
    study_plan = models.ForeignKey(StudyPlan, on_delete=models.CASCADE)
    content = models.TextField(help_text="The full Markdown content of the generated pack.")
    ai_insight = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Power Pack for {self.study_plan.title} ({self.created_at.strftime('%Y-%m-%d')})"
    

# In learning/models.py, add this new model in the SPACED REPETITION section

class RevisionCard(models.Model):
    """
    An AI-generated, concise summary of a lesson for spaced repetition review.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='revision_cards')
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, related_name='revision_cards')
    
    # The AI-generated summary notes in Markdown format
    content = models.TextField()

    # Spaced Repetition Fields
    next_review_date = models.DateField(default=timezone.now)
    last_interval_days = models.IntegerField(default=1)
    
    # A score representing how well the user knows this card
    memory_strength = models.FloatField(default=0.1)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # Ensures a user only has one revision card per lesson to avoid clutter
        unique_together = ('user', 'lesson')

    def __str__(self):
        return f"Revision Card for {self.user.username} on '{self.lesson.title}'"