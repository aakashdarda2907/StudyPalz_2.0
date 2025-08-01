# learning/ai_utils.py
# Django Imports
from django.conf import settings
from django.utils import timezone
from django.contrib.auth.models import User
import json
from django.db import transaction
from django.db.models import Count, Avg


from django.db.models import Count # <-- ADD THIS LINE
import re

"""
AI Utilities for Learning Management System

This module contains all AI-powered functions for content generation,
personalization, and intelligent recommendations using Google's Gemini API.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard Library Imports
from itertools import count
import time
from collections import Counter
from datetime import date, timedelta

# Third-party Imports
import google.generativeai as genai
from googleapiclient.discovery import build

# Django Imports
from django.conf import settings
from django.utils import timezone
from django.contrib.auth.models import User
import json
from django.db import transaction

# Local Model Imports
from .models import (
    Lesson, Mastery, Module, Notification, PowerPack, QuizAttempt, ScheduledDay, ScheduledTask, StudyPlan, UserAnswer, UserInteraction, Quiz, Question, Answer, UserLessonProgress, 
    Flashcard, UserFlashcard
)


# =============================================================================
# API CONFIGURATION
# =============================================================================

# Configure Gemini API
genai.configure(api_key=settings.GEMINI_API_KEY)


def call_gemini_api(prompt):
    """
    Central function to call Gemini API with error handling.
    
    Args:
        prompt (str): The prompt to send to Gemini
        
    Returns:
        str: Generated response or empty JSON on error
    """
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt, request_options={'timeout': 90})
        return response.text
    except Exception as e:
        print(f"An error occurred with the Gemini API: {e}")
        return "{}"  # Return empty JSON on error to prevent crashes


# =============================================================================
# CONTENT GENERATION FUNCTIONS
# =============================================================================

def generate_syllabus_structure(syllabus_text):
    """
    Convert free-form syllabus text into structured JSON format using AI.
    
    Args:
        syllabus_text (str): Raw syllabus input from user
        
    Returns:
        str: JSON string with structured syllabus data
    """
    prompt = (
         "You are an expert curriculum designer AI. Analyze the following syllabus text and convert it into a structured JSON format. "
        "First, determine the main 'subject' of the syllabus (e.g., 'Computer Science', 'Data Science', 'Mathematics', etc etc). "
        "The JSON should be a single object with two keys: 'subject' and 'modules'. The value of 'modules' should be a list of module objects. "
        "Each module object must have 'module_title', 'estimated_hours', 'difficulty', 'prerequisites', and a 'lessons' list. "
        "Each lesson object must have 'lesson_title' and 'keywords'. "
        "CRITICAL: The final output must be ONLY the raw JSON object, starting with `{` and ending with `}`. Do not include any extra text, explanations, or markdown formatting."
        f"\n\nHere is the syllabus:\n---{syllabus_text}---"
    )
    return call_gemini_api(prompt)


def generate_lesson_details(lesson_title):
    """
    Generate comprehensive lesson materials including notes and flowcharts.
    
    Args:
        lesson_title (str): Title of the lesson
        
    Returns:
        str: JSON string with lesson details
    """
    prompt = (
        f"For the lesson titled '{lesson_title}', generate learning materials. "
        "Return a single, raw JSON object with two keys: 'revision_notes' and 'flowchart'. "
        "For 'flowchart', create a simple flowchart with 3 to 5 steps using Mermaid.js 'graph TD' syntax. "
        "**CRITICAL: Use only basic node shapes like `A[Text]` and only the `-->` arrow type.** For example: `graph TD; A[Step 1] --> B[Step 2];`. "
        "Wrap the entire Mermaid syntax block with [MERMAID_START] and [MERMAID_END] tags. "
        "Do not include any text outside of the JSON object."
    )
    return call_gemini_api(prompt)


# In learning/ai_utils.py

def get_single_content_piece(lesson_title, content_type):
    """
    Generate specific, context-aware content for a lesson.
    The 'example' type is now dynamic based on the lesson title.
    """
    content_instructions = {
        'notes': "Generate a set of detailed, point-wise revision notes in Markdown format. Where applicable, include simple diagrams using Mermaid.js syntax (wrapped in [MERMAID_START] and [MERMAID_END] tags) and provide clear examples, especially for concepts involving construction or steps.",
        'analogy': "Generate a simple, easy-to-understand real-world analogy. You can be detailed if needed and be super interesting",
        'example': (
            "You are an expert teacher. Your task is to provide a single, clear, and relevant example to help a student understand the topic. "
            "Analyze the topic and provide the most appropriate type of example: "
            "- If the topic is about programming (e.g., 'loops', 'functions', 'data structures'), provide a well-commented Python, C  and C++ code snippet. "
            "- If the topic is mathematical or scientific (e.g., 'calculus', 'geometry', 'physics'), provide a sample problem and a step-by-step solution. "
            "- If the topic is historical, describe a key event or figure that exemplifies the concept. "
            "- If it's a grammar or language topic, provide a sentence demonstrating the rule clearly. similarly for all other topics"
            "Use Markdown for all formatting."
        ),
        'podcast_script': "You are StudyPalz - an excellent teacher. Rewrite the key concepts of this lesson into a conversational and smooth podcast script suitable for audio playback. Start with a welcoming phrase."
    }
    
    instruction = content_instructions.get(content_type, "")
    if not instruction:
        return ""

    prompt = (
        f"For the lesson titled '{lesson_title}', {instruction} "
        "Do not include any extra text or titles, only the requested content."
    )
    return call_gemini_api(prompt)


def get_generative_ai_explanation(user, lesson, mode, custom_doubt=""):
    """
    Generate personalized AI tutor explanations based on user preferences and mode.
    
    Args:
        user: Django User object
        lesson: Lesson object
        mode (str): Type of explanation requested
        custom_doubt (str): Specific question from user
        
    Returns:
        str: AI-generated explanation
    """
    try:
        learning_style = user.userprofile.get_learning_style_display()
    except:
        learning_style = 'Reading/Writing'

    # Determine prompt instruction based on mode
    mode_instructions = {
        'analogy': "Please explain the core concept using a simple, real-world analogy.",
        'code_example': "Please provide a clear and concise Python code example demonstrating this concept. Use comments to explain each part of the code.",
        'flowchart': "Please describe the process or concept as a flowchart using Mermaid.js graph syntax. **Crucially, you must wrap the entire Mermaid syntax block with [MERMAID_START] on a new line before it and [MERMAID_END] on a new line after it.** Do not use markdown code fences for the mermaid block."
    }

    if custom_doubt:
        prompt_instruction = f"The student has a specific question: '{custom_doubt}'. Please answer this question directly in the context of the lesson."
    elif mode in mode_instructions:
        prompt_instruction = mode_instructions[mode]
    else:
        prompt_instruction = f"Please provide a simple, clear, and encouraging explanation of this topic, tailored to a '{learning_style}' learning style."

    prompt = (
        f"You are StudyPalz, a friendly and helpful AI tutor. A student is studying a lesson titled '{lesson.title}'.\n\n"
        f"{prompt_instruction}\n\n"
        f"Always format your response using Markdown. Here is the lesson content for context:\n\n---\n{lesson.content}\n---\n\n"
    )
    return call_gemini_api(prompt)


# =============================================================================
# YOUTUBE INTEGRATION
# =============================================================================

def find_youtube_video(query):
    """
    Search YouTube for educational videos related to the query.
    
    Args:
        query (str): Search term for YouTube
        
    Returns:
        str: YouTube embed URL or None if not found
    """
    try:
        youtube = build('youtube', 'v3', developerKey=settings.YOUTUBE_API_KEY)
        search_query = f"{query} educational video tutorial explained"

        request = youtube.search().list(
            q=search_query,
            part='snippet',
            maxResults=1,
            type='video'
        )
        response = request.execute()

        if response.get('items'):
            video_id = response['items'][0]['id']['videoId']
            return f'https://www.youtube.com/embed/{video_id}'
            
    except Exception as e:
        print(f"An error occurred with the YouTube API: {e}")
        
    return None


# =============================================================================
# USER PROGRESS & RECOMMENDATIONS
# =============================================================================
# In learning/ai_utils.py, replace this function
def recommend_next_lesson(user):
    """
    AI-powered recommendation system for the next lesson to study.
    """
    # CORRECTED: Filter by lesson object, not concept_key
    weak_mastery = Mastery.objects.filter(
        user=user, 
        mastery_score__lt=0.6,
        lesson__isnull=False
    ).order_by('mastery_score').first()
    
    if weak_mastery:
        return weak_mastery.lesson # Directly return the lesson from the mastery record

    # The rest of the function for logical progression is correct
    last_progress = UserLessonProgress.objects.filter(user=user).order_by('-completed_at').first()
    
    if last_progress:
        last_lesson = last_progress.lesson
        next_lesson = Lesson.objects.filter(
            module=last_lesson.module, 
            order__gt=last_lesson.order
        ).order_by('order').first()
        
        if next_lesson:
            return next_lesson
            
        next_module = Module.objects.filter(
            order__gt=last_lesson.module.order
        ).order_by('order').first()
        
        if next_module:
            return Lesson.objects.filter(module=next_module).order_by('order').first()
    
    return Lesson.objects.all().order_by('module__order', 'order').first()
# In learning/ai_utils.py, replace the existing update_mastery function

# def update_mastery(user, lesson, quiz_score):
#     """
#     Primary entry point for updating a user's mastery.
#     This function now calls the new comprehensive function that includes forgetting curve calculations.
#     """
#     update_mastery_and_forgetting_curve(user=user, lesson=lesson, new_score=quiz_score)
# In learning/ai_utils.py, replace the existing detect_learning_persona function
# In learning/ai_utils.py, replace the existing detect_learning_persona function

def detect_learning_persona(user_interactions):
    """
    Analyzes user interaction patterns, with added weight for positive feedback,
    to classify a user's learning persona.
    """
    personas = {
        'notes': {'name': 'The Scholar', 'animal': '🦉', 'description': 'You seem to learn best by reading detailed notes and text.'},
        'video': {'name': 'The Visualizer', 'animal': '🦅', 'description': 'You seem to learn best by watching videos and seeing concepts in action.'},
        'example': {'name': 'The Practitioner', 'animal': '🐝', 'description': 'You seem to learn best by doing and experimenting with code examples.'},
        'analogy': {'name': 'The Connector', 'animal': '🐬', 'description': 'You seem to learn best by relating new ideas to concepts you already know.'},
        'audio': {'name': 'The Listener', 'animal': '🐺', 'description': 'You seem to learn best by listening to explanations.'},
        'default': {'name': 'The Explorer', 'animal': '🦊', 'description': 'You use a balanced mix of resources. Keep exploring!'}
    }

    if not user_interactions:
        return personas['default']

    # --- NEW: Weighted Score Calculation ---
    interaction_scores = Counter()
    for interaction in user_interactions:
        # Normalize the interaction type by removing prefixes
        interaction_type = interaction.interaction_type.replace('requested_', '').replace('viewed_', '').replace('feedback_', '')
        
        # A simple interaction (like viewing or requesting) gets 1 point
        score = 1
        # A "liked" interaction (feedback=1) gets 2 extra points, making it 3x more valuable
        if interaction.feedback == 1:
            score += 2
        
        interaction_scores[interaction_type] += score

    if not interaction_scores:
        return personas['default']

    # Find the interaction type with the highest total score
    most_frequent_type = interaction_scores.most_common(1)[0][0]
    
    return personas.get(most_frequent_type, personas['default'])


def find_weakness_of_the_week(user):
    """
    Identify the most critical weak topic for focused improvement.
    
    Args:
        user: Django User object
        
    Returns:
        dict: Weakness information with lesson and score, or None
    """
    # Get all mastery records below 70%
    weak_masteries = Mastery.objects.filter(user=user, mastery_score__lt=0.7)

    if not weak_masteries:
        return None  # No significant weaknesses

    best_weakness = None
    max_impact = -1

    for mastery in weak_masteries:
        # Impact score based on how low the mastery is
        # Future versions could calculate prerequisite dependencies
        impact_score = 1 - mastery.mastery_score
        
        if impact_score > max_impact:
            max_impact = impact_score
            lesson = Lesson.objects.filter(concept_key=mastery.concept_key).first()
            if lesson:
                best_weakness = {
                    'lesson': lesson,
                    'score': int(mastery.mastery_score * 100)
                }

    return best_weakness

# In learning/ai_utils.py, replace the whole section

# =============================================================================
# FORGETTING CURVE & MASTERY
# =============================================================================

def update_mastery(user, lesson, quiz_score):
    """
    Primary entry point for updating a user's mastery.
    This function now calls the new comprehensive function and returns the generated AI insight.
    """
    # Pass the call down and return the insight message it generates
    return update_mastery_and_forgetting_curve(user=user, lesson=lesson, new_score=quiz_score)


def update_mastery_and_forgetting_curve(user, lesson, new_score):
    """
    Updates a user's mastery score, recalculates their forgetting curve parameters,
    and returns a user-facing AI insight message about what happened.
    
    Returns:
        str: A user-facing AI insight message.
    """
    if not lesson:
        return None

    mastery, created = Mastery.objects.get_or_create(
        user=user,
        lesson=lesson,
    )

    # --- AI Insight Generation ---
    ai_insight = ""
    previous_score = mastery.mastery_score
    
    # --- Forgetting Curve Calculation ---
    if created:
        mastery.memory_strength = new_score
        days_to_next_review = int(2 * (1 + new_score))
        ai_insight = f"This is your first quiz on '{lesson.title}'! Your initial mastery is {new_score:.0%}. We've scheduled your first review to help lock this in."
    else:
        old_strength = mastery.memory_strength
        last_test = mastery.last_tested_date or timezone.now()
        days_since_last_test = (timezone.now() - last_test).days
        
        decay_factor = (1 - mastery.mastery_score) / max(days_since_last_test, 1)
        performance_gain = (new_score - mastery.mastery_score)
        new_strength = old_strength + performance_gain - decay_factor
        mastery.memory_strength = max(0.1, new_strength)

        days_to_next_review = int(2 * (1 + mastery.memory_strength))

        # Craft the insight based on performance
        if new_score > previous_score:
            ai_insight = f"Great improvement on '{lesson.title}'! Your memory strength has increased. Your next review is scheduled accordingly to keep your knowledge sharp."
        else:
            ai_insight = f"It's normal for memory to fade. Your score on '{lesson.title}' dropped a bit, so I've updated your memory model and scheduled a review sooner to help reinforce it."


    # Update the mastery record with the new data
    mastery.mastery_score = new_score
    mastery.last_tested_date = timezone.now()
    mastery.next_review_date = timezone.now().date() + timedelta(days=max(1, days_to_next_review))
    mastery.save()

    # Return the generated insight
    return ai_insight
# =============================================================================
# GAMIFICATION & MOTIVATION
# =============================================================================
# In learning/ai_utils.py, replace this function
def get_mindset_coach_message(user, lesson, new_score):
    """
    Generate motivational coaching messages based on performance patterns.
    """
    try:
        # CORRECTED: Get mastery by lesson, not concept_key
        previous_mastery = Mastery.objects.get(user=user, lesson=lesson).mastery_score
    except Mastery.DoesNotExist:
        previous_mastery = 0.0

    if previous_mastery < 0.5 and new_score >= 0.8:
        return (f"Amazing persistence on '{lesson.title}'! You went from struggling to mastering it. That's how real progress is made. Keep it up!")
    
    if new_score < 0.4:
        return (f"'{lesson.title}' can be a tough topic. Don't worry about the score, the important thing is to understand the gaps. Review the lesson or ask the AI for a hint!")
    
    return None

# =============================================================================
# SPACED REPETITION SYSTEM
# =============================================================================

# In learning/ai_utils.py, add this new function in the SPACED REPETITION SYSTEM section
# In learning/ai_utils.py, REPLACE this function
def get_daily_flashcard_review_deck(user):
    """
    Constructs a personalized daily flashcard review deck for a user.
    Now includes upcoming cards for the next 7 days.
    """
    today = timezone.now().date()
    
    # 1. Get cards from the SRS queue due today or earlier
    srs_review_cards = UserFlashcard.objects.filter(
        user=user,
        next_review_date__lte=today
    ).select_related('flashcard', 'flashcard__lesson').order_by('next_review_date')

    # 2. Get cards related to topics quizzed on TODAY
    todays_quiz_attempts = QuizAttempt.objects.filter(user=user, timestamp__date=today)
    lessons_quizzed_today_ids = set(ua.question.lesson_id for attempt in todays_quiz_attempts for ua in attempt.useranswer_set.filter(question__lesson__isnull=False))
    
    srs_card_ids = srs_review_cards.values_list('flashcard_id', flat=True)
    todays_topic_cards = UserFlashcard.objects.filter(
        user=user,
        flashcard__lesson_id__in=lessons_quizzed_today_ids
    ).exclude(flashcard_id__in=srs_card_ids).select_related('flashcard', 'flashcard__lesson')

    # 3. NEW: Get upcoming flashcards for the next 7 days
    upcoming_cards = UserFlashcard.objects.filter(
        user=user,
        next_review_date__gt=today,
        next_review_date__lte=today + timedelta(days=7)
    ).select_related('flashcard', 'flashcard__lesson').order_by('next_review_date')

    return {
        'srs_cards': list(srs_review_cards),
        'todays_topic_cards': list(todays_topic_cards),
        'upcoming_cards': list(upcoming_cards),
    }

def generate_flashcards_for_lesson(user, lesson):
    """
    Auto-generate flashcards when a lesson is completed.
    
    Args:
        user: Django User object
        lesson: Lesson object
    """
    question = f"What is the main concept of '{lesson.title}'?"
    answer = lesson.content[:500] + "..." if len(lesson.content) > 500 else lesson.content
    
    flashcard, _ = Flashcard.objects.get_or_create(
        lesson=lesson, 
        question=question, 
        defaults={'answer': answer}
    )
    UserFlashcard.objects.get_or_create(user=user, flashcard=flashcard)


def update_flashcard_review(user_flashcard, recalled_correctly):
    """
    Update flashcard review schedule based on spaced repetition algorithm.
    
    Args:
        user_flashcard: UserFlashcard object
        recalled_correctly (bool): Whether user recalled correctly
    """
    if recalled_correctly:
        # Double the interval for correct answers
        new_interval = user_flashcard.last_interval_days * 2
    else:
        # Reset to 1 day for incorrect answers
        new_interval = 1
        
    user_flashcard.last_interval_days = new_interval
    user_flashcard.next_review_date = timezone.now().date() + timedelta(days=new_interval)
    user_flashcard.save()


# =============================================================================
# SCHEDULING & PLANNING
# =============================================================================

def create_study_schedule(study_plan, exam_date):
    """
    Create a basic day-by-day study schedule.
    
    Args:
        study_plan: StudyPlan object
        exam_date: Target exam date
        
    Returns:
        list: Schedule with dates and topics
    """
    today = date.today()
    days_available = (exam_date - today).days

    if days_available <= 0:
        return []

    lessons = Lesson.objects.filter(
        module__study_plan=study_plan
    ).order_by('module__order', 'order')
    total_lessons = lessons.count()

    if total_lessons == 0:
        return []

    # Distribute lessons over available days
    lessons_per_day = total_lessons / days_available
    
    schedule = []
    lesson_index = 0
    
    for day_num in range(days_available):
        current_date = today + timedelta(days=day_num)
        day_of_week = current_date.strftime('%A')
        
        start_index = int(lesson_index)
        end_index = int(lesson_index + lessons_per_day)
        
        lessons_for_the_day = lessons[start_index:end_index]
        
        if lessons_for_the_day:
            topic_str = ", ".join([l.title for l in lessons_for_the_day])
            schedule.append({
                'date': current_date.strftime('%B %d, %Y'),
                'day_of_week': day_of_week,
                'topic': topic_str,
            })
        
        lesson_index += lessons_per_day

    return schedule


def create_advanced_schedule(study_plan, exam_date, module_ids, backup_days=0):
    """
    Create an intelligent study schedule with even distribution and backup days.
    
    Args:
        study_plan: StudyPlan object
        exam_date: Target exam date
        module_ids: List of selected module IDs
        backup_days (int): Number of backup/review days
        
    Returns:
        list: Detailed schedule with tasks and dates
    """
    today = date.today()
    total_days = (exam_date - today).days
    study_days = total_days - int(backup_days)

    if study_days <= 0:
        return []

    # Get selected modules and their lessons
    modules = study_plan.module_set.filter(id__in=module_ids).order_by('order')
    lessons = list(Lesson.objects.filter(
        module__in=modules
    ).order_by('module__order', 'order'))
    
    if not lessons:
        return []

    total_lessons = len(lessons)
    schedule = []
    
    # Evenly distribute lessons across study days
    for i, lesson in enumerate(lessons):
        day_num = int(i * (study_days / total_lessons))
        schedule_date = today + timedelta(days=day_num)
        
        schedule.append({
            'date': schedule_date,
            'task_type': 'STUDY',
            'description': f"Study: {lesson.title}",
            'lesson': lesson
        })

    # Add backup/review days
    for i in range(int(backup_days)):
        schedule_date = today + timedelta(days=study_days + i)
        schedule.append({
            'date': schedule_date,
            'task_type': 'REVIEW',
            'description': 'Backup / Revision Day',
            'lesson': None
        })
        
    return schedule


# =============================================================================
# ANALYTICS & INSIGHTS
# =============================================================================

# In learning/ai_utils.py, add this new function in the ANALYTICS & INSIGHTS section
# In learning/ai_utils.py, add this new function in the ANALYTICS & INSIGHTS section

# In learning/ai_utils.py
# In learning/ai_utils.py

def get_user_analytics_profile(user):
    """
    Gathers and analyzes all of a user's data to create a comprehensive,
    deep analytics profile for the "My Progress" page. This upgraded version
    categorizes topics by mastery level and includes historical quiz data.
    """
    profile = {}
    
    # 1. Calculate Syllabus Completion Percentage (based on passed lesson quizzes)
    latest_plan = StudyPlan.objects.filter(user=user).order_by('-created_at').first()
    if latest_plan:
        total_lessons_in_plan = Lesson.objects.filter(module__study_plan=latest_plan).count()
        completed_lessons_count = UserLessonProgress.objects.filter(
            user=user, 
            lesson__module__study_plan=latest_plan
        ).count()
        profile['progress_percent'] = int((completed_lessons_count / total_lessons_in_plan) * 100) if total_lessons_in_plan > 0 else 0
    else:
        profile['progress_percent'] = 0
        total_lessons_in_plan = 0

    # 2. Determine Learning Persona (existing logic is sound)
    all_interactions = UserInteraction.objects.filter(user=user)
    profile['learning_persona'] = detect_learning_persona(all_interactions)

    # 3. Granular Mastery Topic Categorization
    all_masteries = Mastery.objects.filter(user=user, lesson__isnull=False).select_related('lesson')
    
    mastered_topics = []
    in_progress_topics = []
    
    for mastery in all_masteries:
        if mastery.mastery_score >= 0.8:
            mastered_topics.append(mastery)
        elif mastery.mastery_score > 0:
            in_progress_topics.append(mastery)
            
    profile['mastered_topics'] = mastered_topics
    profile['in_progress_topics'] = sorted(in_progress_topics, key=lambda m: m.mastery_score)

    # Find topics the user hasn't been quizzed on at all
    if latest_plan:
        mastered_lesson_ids = {m.lesson.id for m in all_masteries}
        unseen_lessons = Lesson.objects.filter(
            module__study_plan=latest_plan
        ).exclude(id__in=mastered_lesson_ids).order_by('module__order', 'order')
        profile['unseen_topics'] = list(unseen_lessons)
    else:
        profile['unseen_topics'] = []

    # 4. Analyze Common Mistake Categories (existing logic is sound)
    incorrect_answers = UserAnswer.objects.filter(
        quiz_attempt__user=user, 
        is_correct=False
    ).values_list('question__category', flat=True)
    
    mistake_counts = Counter(incorrect_answers)
    if mistake_counts:
        profile['common_mistake'] = mistake_counts.most_common(1)[0][0]
    else:
        profile['common_mistake'] = "None yet!"

    # 5. NEW: Gather Quiz Performance History for Chart
    quiz_history = QuizAttempt.objects.filter(user=user).order_by('timestamp')
    history_data = []
    for attempt in quiz_history:
        # Calculate the overall percentage score for the attempt
        # This handles both multiple-choice and AI-graded open-ended questions
        total_score = sum(ua.ai_score if ua.question.question_type == 'open-ended' and ua.ai_score is not None else (1 if ua.is_correct else 0) for ua in attempt.useranswer_set.all())
        num_questions = attempt.total_questions
        percent = round((total_score / num_questions) * 100) if num_questions > 0 else 0
        history_data.append({
            'date': attempt.timestamp.strftime('%b %d, %Y'),
            'score': percent,
            'quiz_title': attempt.quiz.title
        })
    profile['quiz_history_chart_data'] = history_data

    return profile

def get_heatmap_data(user):
    """
    Generate heatmap data for user activity visualization.
    
    Args:
        user: Django User object
        
    Returns:
        dict: Timestamp-count pairs for Cal-Heatmap
    """
    ninety_days_ago = timezone.now() - timedelta(days=90)
    interactions = UserInteraction.objects.filter(
        user=user, 
        timestamp__gte=ninety_days_ago
    )

    # Count interactions per day
    daily_counts = Counter(inter.timestamp.date() for inter in interactions)

    # Convert to dictionary with timestamps in milliseconds for Cal-Heatmap v4
    heatmap_data = {
        int(time.mktime(day.timetuple())) * 1000: count
        for day, count in daily_counts.items()
    }

    return heatmap_data


# =============================================================================
# SOCIAL & COMMUNITY FEATURES
# =============================================================================

def find_study_pals(user):
    """
    Find study partners and mentors based on complementary skill levels.
    
    Args:
        user: Django User object
        
    Returns:
        tuple: (peers_list, mentors_list, weak_concept_key)
    """
    # Find user's weakest concept
    weakest_mastery = Mastery.objects.filter(
        user=user, 
        mastery_score__lt=0.7
    ).order_by('mastery_score').first()
    
    if not weakest_mastery:
        return [], [], None

    weak_concept_key = weakest_mastery.concept_key
    
    # Find peers with similar struggles
    peers_masteries = Mastery.objects.filter(
        concept_key=weak_concept_key, 
        mastery_score__lt=0.7
    ).exclude(user=user)
    
    # Find potential mentors who have mastered this concept
    mentors_masteries = Mastery.objects.filter(
        concept_key=weak_concept_key, 
        mastery_score__gte=0.9
    ).exclude(user=user)
    
    peers = [mastery.user for mastery in peers_masteries]
    mentors = [mastery.user for mastery in mentors_masteries]
    
    return peers, mentors, weak_concept_key


def generate_personalized_project(user):
    """
    Generate project ideas based on user interests and progress.
    
    Args:
        user: Django User object
        
    Returns:
        dict: Project idea with title, description, and dataset
    """
    try:
        user_interests = [
            i.strip().lower() 
            for i in user.userprofile.interests.split(',')
        ]
        primary_interest = user_interests[0] if user_interests else 'general'
    except:
        primary_interest = 'general'
    
    # Mock project ideas by domain
    mock_project_ideas = {
        'finance': {
            'title': 'Stock Price Trend Analyzer',
            'description': 'Build a machine learning model to predict stock price movements using historical data and technical indicators.',
            'dataset': 'Yahoo Finance API or Quandl financial datasets'
        },
        'healthcare': {
            'title': 'Heart Disease Risk Predictor',
            'description': 'Create a classification model to assess heart disease risk based on patient health metrics.',
            'dataset': 'UCI Heart Disease dataset'
        },
        'sports': {
            'title': 'Basketball Shot Success Predictor',
            'description': 'Analyze basketball shooting data to predict shot success probability based on various factors.',
            'dataset': 'NBA shot chart data from stats.nba.com'
        },
        'general': {
            'title': 'Customer Churn Prediction',
            'description': 'Build a model to predict which customers are likely to leave a service based on usage patterns.',
            'dataset': 'Telco Customer Churn dataset from Kaggle'
        }
    }
    
    return mock_project_ideas.get(primary_interest, mock_project_ideas['general'])


def get_job_market_skills():
    """
    Get current job market demand for different skills (mock data).
    
    Returns:
        dict: Skill names mapped to demand percentages
    """
    return {
        'intro_to_python': 95,
        'linear_regression': 75,
        'what_is_ds': 88,
        'sql_basics': 85,
        'data_visualization': 70,
        'deep_learning': 45,
        'natural_language_processing': 40,
    }


#Phase 4 
# learning/ai_utils.py
# In learning/ai_utils.py
# In learning/ai_utils.py
from collections import Counter # Make sure Counter is imported

# ... (other functions)
# In learning/ai_utils.py
from collections import Counter # Make sure Counter is imported at the top of the file

# ... (all other functions remain the same) ...
# In learning/ai_utils.py
# In learning/ai_utils.py
from collections import Counter # Make sure Counter is imported

# ... (other functions)

def generate_quiz_questions(topics, num_questions, question_type, difficulty_profile):
    """
    Generates a personalized quiz that can include a mix of question types
    based on a detailed difficulty profile.
    """
    difficulty = difficulty_profile.get('difficulty', 'Medium')
    mastery_score = difficulty_profile.get('mastery_score', 'N/A')
    weaknesses = difficulty_profile.get('weaknesses', {})

    instruction = f"The user's determined difficulty level is '{difficulty}'. Their mastery score is {mastery_score}."

    if weaknesses:
        weakness_summary = ", ".join([f"{cat} ({num} wrong)" for cat, num in weaknesses.items()])
        instruction += f" They have shown specific weakness in: {weakness_summary}. Prioritize questions that test these categories."

    else:
        # --- ADD THIS LINE ---
        instruction += " Create a balanced mix of question categories (CONCEPTUAL, CODE, TERMINOLOGY)."

    # --- NEW: Instructions for question types ---
    if question_type == 'mixed':
        type_instruction = "Generate a mix of 'multiple-choice' and 'open-ended' questions. Aim for about half of each."
    elif question_type == 'open-ended':
        type_instruction = "All questions must be 'open-ended'."
    else:
        type_instruction = "All questions must be 'multiple-choice'."

    prompt = f"""
    You are an expert quiz creation assistant. Your task is to generate a {num_questions}-question quiz about: {', '.join(topics)}.
    
    {type_instruction}

    You must create a personalized quiz based on the following user profile:
    ---
    {instruction}
    ---

    IMPORTANT: Your entire response must be a single, valid JSON object. Do not include any other text.
    The JSON structure must be an object with a "questions" key, which is a list of question objects.
    
    For a "multiple-choice" question, the structure must be:
    {{
      "topic": "...",
      "question_type": "multiple-choice",
      "question_text": "...",
      "answers": [ "...", "...", "...", "..." ],
      "correct_answer": <index>,
      "category": "..."
    }}

    For an "open-ended" question, the structure must be:
    {{
      "topic": "...",
      "question_type": "open-ended",
      "question_text": "...",
      "grading_rubric": "A brief markdown rubric explaining what a correct answer should contain.",
      "category": "..."
    }}
    """
    return call_gemini_api(prompt)
# learning/ai_utils.py

# learning/ai_utils.py
# In learning/ai_utils.py

def analyze_quiz_and_reschedule(quiz_attempt):
    user = quiz_attempt.user
    
    # Find all incorrect answers for this attempt that have a lesson linked to the question
    incorrect_answers = UserAnswer.objects.filter(
        quiz_attempt=quiz_attempt, 
        is_correct=False,
        question__lesson__isnull=False
    ).select_related('question__lesson')

    # Find the unique set of lessons where the user made mistakes
    weak_lessons = {user_answer.question.lesson for user_answer in incorrect_answers}

    if not weak_lessons:
        return # No review needed if all incorrect answers were on general questions

    latest_plan = StudyPlan.objects.filter(user=user).order_by('-created_at').first()
    if not latest_plan:
        return

    # Schedule the review for the next day
    tomorrow = timezone.localdate() + timedelta(days=1)
    day, _ = ScheduledDay.objects.get_or_create(study_plan=latest_plan, date=tomorrow)

    # Create a separate, correctly linked review task for each weak lesson
    for lesson in weak_lessons:
        ScheduledTask.objects.create(
            scheduled_day=day,
            task_type='REVIEW',
            # --- THIS IS THE KEY FIX ---
            lesson=lesson, 
            task_description=f"Review: {lesson.title}",
            triggering_attempt=quiz_attempt
        )

# learning/ai_utils.py

def generate_targeted_review_notes(lesson_title, weak_category):
    """
    Generates revision notes specifically targeting a user's weak category for a lesson.
    """
    prompt = (
        f"A student is struggling with '{weak_category}' questions for the lesson titled '{lesson_title}'. "
        "Your task is to generate a set of highly targeted revision notes that focus specifically on fixing this weakness. "
        "Use clear headings, bullet points, and bold text in Markdown format. "
        "Provide only the notes, with no extra conversational text."
    )
    return call_gemini_api(prompt)

# learning/ai_utils.py
def get_tutor_response(lesson_content, user_question):
    """
    Acts as a real-time tutor, answering a specific user question
    based on the context of the lesson content.
    """
    prompt = (
        f"You are StudyPalz, a friendly and expert AI tutor. A student is studying a lesson with the following content:\n\n"
        f"--- LESSON CONTENT ---\n{lesson_content}\n---------------------\n\n"
        f"The student has a specific question: '{user_question}'\n\n"
        "Please provide a clear, simple, and direct answer to the student's question based on the provided lesson content. Add some bold words and keep it pointwise "
        "Format your response in Markdown."
    )
    return call_gemini_api(prompt)
# In learning/ai_utils.py

# ... (keep all other existing functions)
# In learning/ai_utils.py

from .models import UserAnswer, Lesson # Make sure to import UserAnswer and Lesson

# ... (keep all other existing functions)
from collections import defaultdict
from .models import UserAnswer, Lesson # Make sure these are imported at the top

# ... (all other functions)

def find_knowledge_gaps(user):
    """
    Analyzes all of a user's answers to identify weak lessons based on a performance threshold.
    This new version correctly handles both multiple-choice and AI-graded open-ended answers.
    """
    all_user_answers = UserAnswer.objects.filter(
        quiz_attempt__user=user, 
        question__lesson__isnull=False
    ).select_related('question__lesson', 'question')

    if not all_user_answers:
        return None

    # Use a more accurate calculation that handles nuanced AI scores
    lesson_performance = defaultdict(lambda: {'score_sum': 0.0, 'count': 0})

    for answer in all_user_answers:
        lesson = answer.question.lesson
        lesson_performance[lesson]['count'] += 1
        
        if answer.question.question_type == 'open-ended':
            # Use the AI's nuanced score for open-ended questions
            lesson_performance[lesson]['score_sum'] += answer.ai_score or 0.0
        else:
            # Use a binary 1 or 0 for multiple-choice questions
            if answer.is_correct:
                lesson_performance[lesson]['score_sum'] += 1

    weak_lessons = []
    for lesson, data in lesson_performance.items():
        # Calculate the average score (accuracy) for the lesson
        accuracy = (data['score_sum'] / data['count']) * 100 if data['count'] > 0 else 0
        if accuracy < 70:
            weak_lessons.append(lesson.title)

    if not weak_lessons:
        return None

    # The rest of the function remains the same
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    A student has shown weakness in the following topics based on their quiz scores (average performance below 70%): {', '.join(weak_lessons)}.

    Please provide a brief, encouraging analysis of these potential knowledge gaps.
    Structure your response with clear headings for each weak topic.
    For each topic, suggest a specific action for improvement, such as "Try generating a new set of Revision Notes" or "Create an Analogy to simplify the concept."
    Keep the tone positive and action-oriented.
    """
    response = model.generate_content(prompt)

    return {
        'weak_topics': weak_lessons,
        'ai_summary': response.text
    }
# In learning/ai_utils.py

def generate_cheat_sheet(topic_title):
    """
    Generates a concise cheat sheet for a given topic.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    You are an academic assistant. Your task is to create a concise "cheat sheet" for the topic: "{topic_title}".

    The cheat sheet should be very brief and focused on the absolute key information. Include:
    - Core definitions.
    - Key formulas or functions.
    - Some simple examples.
    - most imp this is for a person who has given a quiz and has scored badly in this topic so make sure the notes are point wise and such that he understands easily cover all imp topics and bold any formula syntax etc
    
    Use markdown for formatting, especially headings and bullet points.
    """
    response = model.generate_content(prompt)
    return response.text

# In learning/ai_utils.py
from .models import UserInteraction # Make sure UserInteraction is imported

# ... (keep all other functions)
# In learning/ai_utils.py
# In learning/ai_utils.py
import json

# ... (other imports at the top of the file) ...
# In learning/ai_utils.py
import json
from collections import Counter
from .models import UserInteraction, Mastery, UserAnswer # Make sure these are imported

# ... (keep all other functions)
# In learning/ai_utils.py, replace this function
# In learning/ai_utils.py, replace this entire function

def get_content_recommendation(user, lesson):
    """
    Recommends the best content type by providing a more robust prompt to the AI,
    ensuring valid JSON is returned even with detailed, multi-line reasons.
    """
    helpful_interactions = UserInteraction.objects.filter(user=user, feedback=1, interaction_type__startswith='requested_')
    interaction_counts = Counter(interaction.interaction_type.replace('requested_', '') for interaction in helpful_interactions)
    history_summary = ", ".join([f"{count} positive feedback on '{name}'" for name, count in interaction_counts.items()]) or "No helpful feedback history yet."

    mastery_score = "Not yet taken a quiz for this lesson."
    try:
        mastery = Mastery.objects.get(user=user, lesson=lesson)
        mastery_score = f"{mastery.mastery_score:.0%}"
    except Mastery.DoesNotExist:
        pass

    incorrect_answers = UserAnswer.objects.filter(
        quiz_attempt__user=user, question__lesson=lesson, is_correct=False
    ).values_list('question__category', flat=True)
    
    category_fails = Counter(incorrect_answers)
    weakness_summary = ", ".join([f"struggles with {cat} questions ({num} wrong)" for cat, num in category_fails.items()]) or "No specific quiz weaknesses identified for this lesson."

    # --- NEW, MORE ROBUST PROMPT ---
    prompt = f"""
    You are an intelligent recommendation engine for a learning platform.
    Your task is to analyze user data for the lesson "{lesson.title}" and recommend the single best content type from the options: "notes", "analogy", or "example".

    USER DATA:
    - Mastery Score: {mastery_score}
    - Quiz Weaknesses: {weakness_summary}
    - Historical Preference: {history_summary}

    YOUR LOGIC:
    1.  Prioritize weaknesses. If the user struggles with 'CODE' or 'CONCEPTUAL' questions, strongly recommend an 'example' or 'analogy'.
    2.  Consider the lesson title. For abstract topics, an 'analogy' is best. for technical topics, an 'example' is best.
    3.  Use historical preference as a tie-breaker.

    YOUR RESPONSE:
    You MUST return a single, valid JSON object and nothing else.
    The "reason" must be a single JSON string, at least 3 lines long, using '\\n' for new lines.

    EXAMPLE RESPONSE FORMAT:
    {{
      "recommendation": "example",
      "reason": "Based on your quiz history for '{lesson.title}', you seem to struggle with code-based questions.\\nThis 'example' will provide a practical, hands-on demonstration to help solidify the concept.\\nFocusing on application is the most effective way for you to improve your mastery of this topic."
    }}
    """
    
    try:
        response_text = call_gemini_api(prompt)
        # It's safer to find the JSON block in case the AI adds any text before/after
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON object found in AI response.")
        
        cleaned_json = json_match.group()
        data = json.loads(cleaned_json)
        
        valid_recs = ["notes", "analogy", "example"]
        if data.get("recommendation") in valid_recs and data.get("reason"):
            return data
            
    except Exception as e:
        print(f"Error processing AI recommendation: {e}\\nResponse text was: {response_text}")
        
    # This fallback will now be used less often
    return {"recommendation": "notes", "reason": "Suggesting notes as a good starting point. Take a quiz to get more personalized recommendations!"}
# In learning/ai_utils.py
import json # Make sure json is imported

# ... (other functions)
# In learning/ai_utils.py

# learning/ai_utils.py
# ... (other imports) ...

def generate_adaptive_mock_exam(user_profile, question_type='mixed', num_questions=20):
    """
    Generates a personalized mock exam that can include a mix of question types
    based on a detailed user knowledge profile.
    """
    profile_json_str = json.dumps(user_profile, indent=2)

    if question_type == 'mixed':
        type_instruction = "Generate a mix of 'multiple-choice' and 'open-ended' questions. Aim for about half of each."
    elif question_type == 'open-ended':
        type_instruction = "All questions must be 'open-ended'."
    else: # Default to multiple-choice
        type_instruction = "All questions must be 'multiple-choice'."

    prompt = f"""
    You are an expert exam creator. Your task is to generate a personalized, {num_questions}-question mock exam.
    
    {type_instruction}

    You must create the exam based on the following detailed user knowledge profile:
    --- USER KNOWLEDGE PROFILE ---
    {profile_json_str}
    ------------------------------

    Your mission is to strategically test the user's weaknesses. Follow these critical instructions:
    1.  **Allocate More Questions to Weak Topics:** Assign more questions to lessons where the 'mastery_score' is low.
    2.  **Scale Difficulty per Topic:** For lessons with high mastery, ask 'Hard' questions. For lessons with low mastery, ask 'Easy' or 'Medium' questions.
    3.  **Target Specific Weaknesses:** If a lesson profile lists 'weak_categories', you MUST create questions of that category for that lesson.

    Return a single, valid JSON object with a "questions" key. For each question, include the "question_type" key.
    
    **CRITICAL:** The entire response must be ONLY the raw JSON object, starting with '{{' and ending with '}}'. Do not include any text before or after the JSON.
    """
    return call_gemini_api(prompt)

# In learning/ai_utils.py
import json # Make sure json is imported at the top of the file

# ... (all your other existing functions)

def grade_open_ended_answer(question_text, rubric, user_answer):
    """
    Uses AI to grade a user's open-ended answer against a rubric.
    Returns a score and constructive feedback.
    """
    prompt = f"""
    You are an AI Teaching Assistant. Your task is to grade a student's answer to an open-ended question.

    - **Question:** "{question_text}"
    - **Grading Rubric:** "A good answer should: {rubric}"
    - **Student's Answer:** "{user_answer}"

    Analyze the student's answer based on the rubric. Determine a score between 0.0 (completely wrong) and 1.0 (perfect).
    Then, provide one sentence of constructive feedback explaining your reasoning for the score. The feedback should be encouraging.

    Return your analysis as a single, valid JSON object with two keys: "score" (a float) and "feedback" (a string).

    Example Response:
    {{
      "score": 0.85,
      "feedback": "This is a strong answer that correctly identifies the main concept, but could be improved by providing a specific example."
    }}
    """
    
    response_text = call_gemini_api(prompt)
    try:
        # Clean up potential markdown code block formatting
        cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
        data = json.loads(cleaned_text)
        
        # Validate data types
        if isinstance(data.get('score'), (int, float)) and isinstance(data.get('feedback'), str):
            # Clamp score between 0.0 and 1.0
            data['score'] = max(0.0, min(1.0, data['score']))
            return data
    except (json.JSONDecodeError, KeyError, Exception):
        pass # Fallback on error

    return {"score": 0.0, "feedback": "Could not automatically grade this answer."}

# In learning/ai_utils.py, at the end of the file

# =============================================================================
# AUTOMATED QUIZZING
# =============================================================================

def generate_automated_quiz(user, quiz_type, lessons_for_quiz):
    """
    Generates a personalized quiz (Daily, Weekly, Surprise) for a user based on specific lessons.
    This function is designed to integrate with the existing ai_utils structure.

    Args:
        user (User): The user for whom the quiz is generated.
        quiz_type (str): 'daily', 'weekly', or 'surprise'.
        lessons_for_quiz (QuerySet<Lesson>): The lessons to be included in the quiz.

    Returns:
        Quiz: The newly created Quiz object, or None if generation fails.
    """
    if not lessons_for_quiz:
        print("Quiz generation skipped: No lessons provided.")
        return None

    # 1. Prepare the content for the AI prompt
    full_content = ""
    lesson_titles = [lesson.title for lesson in lessons_for_quiz]
    for lesson in lessons_for_quiz:
        full_content += f"--- START OF CONTENT FOR '{lesson.title}' ---\n"
        # Use the 'content' field from your Lesson model
        full_content += lesson.content or ""
        full_content += "\n--- END OF CONTENT ---\n\n"

    # 2. Craft the AI Insight and Title based on the quiz type
    ai_insight = ""
    quiz_title = ""
    if quiz_type == 'daily':
        quiz_title = "Daily Review Quiz"
        ai_insight = f"This is your daily review quiz. It covers the topics you studied recently ({', '.join(lesson_titles)}) to help reinforce your memory."
    elif quiz_type == 'weekly':
        quiz_title = "Weekly Progress Check"
        ai_insight = f"This is your weekly progress quiz, focusing on these topics: {', '.join(lesson_titles)}. We've selected these to ensure you have a balanced understanding across your study plan."
    elif quiz_type == 'surprise':
        quiz_title = "Surprise Knowledge Test!"
        ai_insight = f"Surprise! This quiz focuses on topics like {', '.join(lesson_titles)}. Your mastery scores here were a bit lower, so this is a great chance to strengthen them."

    # 3. Create the main prompt for the Gemini API
    prompt = f"""
    You are an expert quiz creator for a learning platform.
    Based on the following content from the lessons titled "{', '.join(lesson_titles)}", generate a 5-question quiz.
    The quiz should include a mix of "multiple-choice" and "open-ended" questions.
    For each question, clearly indicate which lesson it pertains to.

    The entire output MUST be a single, valid JSON object with a key "questions", which is a list of question objects.
    Each question object must have the following keys:
    - "lesson_title": The exact title of the lesson this question is from.
    - "question_type": Either "multiple-choice" or "open-ended".
    - "category": Choose the best category from 'CONCEPTUAL', 'CODE', or 'TERMINOLOGY'.
    - "text": The full text of the question.
    - "answers": For "multiple-choice", a list of objects, each with "text" and "is_correct" (boolean). One must be true.
    - "grading_rubric": For "open-ended", a brief rubric.
    - "explanation": (Optional) A brief explanation for the correct answer.

    Content to use for quiz generation:
    {full_content}
    """

    try:
        # Use the existing central function to call the API
        response_text = call_gemini_api(prompt)
        
        # Clean up potential markdown code block formatting
        cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
        quiz_data = json.loads(cleaned_text)

        # 4. Create the Quiz and associated questions in the database atomically
        with transaction.atomic():
            new_quiz = Quiz.objects.create(
                user=user,
                title=quiz_title,
                quiz_type=quiz_type,
                ai_insight=ai_insight
            )

            for q_data in quiz_data.get('questions', []):
                lesson_title = q_data.get('lesson_title')
                # Find the correct lesson object based on the title and user's plan
                lesson_obj = Lesson.objects.filter(title__iexact=lesson_title, module__study_plan__user=user).first()

                question = Question.objects.create(
                    quiz=new_quiz,
                    lesson=lesson_obj,
                    text=q_data.get('text'), # Matches your 'Question' model
                    question_type=q_data.get('question_type'), # Matches your 'Question' model
                    category=q_data.get('category', 'CONCEPTUAL'), # Matches your 'Question' model
                    grading_rubric=q_data.get('grading_rubric') # Matches your 'Question' model
                )

                if q_data.get('question_type') == 'multiple-choice':
                    for ans_data in q_data.get('answers', []):
                        Answer.objects.create(
                            question=question,
                            text=ans_data.get('text'), # Matches your 'Answer' model
                            is_correct=ans_data.get('is_correct', False) # Matches your 'Answer' model
                        )
        return new_quiz

    except Exception as e:
        print(f"Error in generate_automated_quiz: {e}")
        return None
    

# In learning/ai_utils.py, add this new section at the end

# =============================================================================
# SCHEDULING & MAINTENANCE
# =============================================================================
# In learning/ai_utils.py, replace the perform_daily_schedule_maintenance function
# In# In learning/ai_utils.py, replace the entire function
# In learning/ai_utils.py, replace the existing function

def perform_daily_schedule_maintenance(user):
    """
    Runs as a nightly process for a user to maintain their study schedule.
    This upgraded version aggressively schedules reviews for weak topics and
    generates targeted flashcards.
    """
    today = timezone.localdate()
    
    latest_plan = StudyPlan.objects.filter(user=user).order_by('-created_at').first()
    if not latest_plan:
        return

    # --- 1. Proactive REVIEW Scheduling for WEAK Topics (New Logic) ---
    # Find all topics with a mastery score below 60% that don't already have a review scheduled for tomorrow.
    review_date = today + timedelta(days=1)
    
    # Get IDs of lessons already scheduled for review tomorrow to avoid duplicates
    existing_review_lesson_ids = ScheduledTask.objects.filter(
        scheduled_day__study_plan=latest_plan,
        scheduled_day__date=review_date,
        task_type='REVIEW'
    ).values_list('lesson_id', flat=True)

    weak_masteries = Mastery.objects.filter(
        user=user,
        mastery_score__lt=0.6
    ).exclude(lesson_id__in=existing_review_lesson_ids)

    if weak_masteries.exists():
        day, _ = ScheduledDay.objects.get_or_create(study_plan=latest_plan, date=review_date)
        
        for mastery in weak_masteries:
            lesson = mastery.lesson
            if not lesson:
                continue

            # a) Create the review task in the schedule
            ScheduledTask.objects.create(
                scheduled_day=day,
                lesson=lesson,
                task_type='REVIEW',
                task_description=f"AI Review: Focus on {lesson.title}"
            )

            # b) Generate new, targeted flashcards for this weak topic
            generate_targeted_flashcards_for_review(user, lesson)

            # c) Create a helpful notification for the user
            Notification.objects.create(
                user=user,
                message=(
                    f"AI Insight: Your mastery of '{lesson.title}' is low ({mastery.mastery_score:.0%}). "
                    f"I've added a review session to your schedule for tomorrow and generated new flashcards to help you practice."
                )
            )

    # --- 2. Dynamic Rescheduling of MISSED Tasks (Existing Logic) ---
    yesterday = today - timedelta(days=1)
    missed_tasks = ScheduledTask.objects.filter(
        scheduled_day__study_plan=latest_plan,
        scheduled_day__date=yesterday,
        completed=False
    )
    
    for task in missed_tasks:
        reschedule_date = today + timedelta(days=3)
        day, _ = ScheduledDay.objects.get_or_create(study_plan=latest_plan, date=reschedule_date)
        
        ScheduledTask.objects.create(
            scheduled_day=day,
            lesson=task.lesson,
            task_type='REVIEW',
            task_description=f"[RESCHEDULED] {task.lesson.title if task.lesson else task.task_description}"
        )
        task.task_description = f"MISSED: {task.task_description}"
        task.save()
        
        if task.lesson:
            Notification.objects.create(
                user=user,
                message=f"AI Insight: You missed '{task.lesson.title}' yesterday, so I've rescheduled it for you in a few days."
            )

# In learning/ai_utils.py, add this new function at the end of the file
# In learning/ai_utils.py, replace the generate_pre_exam_power_pack function
# In learning/ai_utils.py, replace the entire generate_pre_exam_power_pack function
# In learning/ai_utils.py, replace the entire generate_pre_exam_power_pack function
# In learning/ai_utils.py, replace this function
# In learning/ai_utils.py, replace this entire function

def generate_pre_exam_power_pack(user, study_plan):
    """
    Generates and SAVES a comprehensive, deeply personalized Pre-Exam Power Pack.
    """
    all_lessons = Lesson.objects.filter(module__study_plan=study_plan).order_by('module__order', 'order')
    if not all_lessons:
        return None

    weak_masteries = Mastery.objects.filter(user=user, lesson__in=all_lessons, mastery_score__lt=0.7).order_by('mastery_score')
    weak_topics = [mastery.lesson.title for mastery in weak_masteries]

    # --- THIS IS THE FIX ---
    # Instead of summarizing the interactions into a dictionary here,
    # we now pass the full objects to the persona detection function.
    all_user_interactions = UserInteraction.objects.filter(user=user)
    persona = detect_learning_persona(all_user_interactions)
    # --- END OF FIX ---

    full_syllabus_content = ""
    for lesson in all_lessons:
        full_syllabus_content += f"--- CONTENT FOR: {lesson.title} ---\n{lesson.content}\n\n"

    prompt = f"""
    You are an expert AI tutor creating a definitive Pre-Exam Power Pack for a student.
    The student's learning style profile suggests they are a '{persona.get('name')}' who learns best by '{persona.get('description')}'.
    The student's weakest topics are: {', '.join(weak_topics) if weak_topics else 'None identified'}.

    Your task is to generate a comprehensive study guide in Markdown format with two distinct parts:

    ### PART 1: UNIVERSAL REVISION GUIDE
    Go through the syllabus lesson by lesson. For EACH lesson, generate NEUTRAL, point-wise revision notes formatted like a cheat sheet. This part is for quick, last-minute review of all topics. It should be neatly formatted with headings, bullet points, and highlight important formulas or concepts.

    ### PART 2: PERSONALIZED "DEEP-DIVE" SECTION
    After the universal guide, create a special section titled "### Personalized Deep-Dive".
    For ALL of the identified weak topics ({', '.join(weak_topics)}), provide an in-depth explanatory tutorial. You MUST tailor the tone and format of THIS section to the student's learning style. For example, for a 'Practitioner', include many code examples. For a 'Visualizer', use Mermaid.js diagrams. For a 'Connector', use analogies.
 
    Dont format all things to code format -- use code only whereever needed - rest all things should be neatly presented in headings,points bold and italics where needed- no need of flow charts. Dont make it too long or too short too -- mention neatly before we start Deep Dive Section.
    --- REFERENCE CONTENT ---
    {full_syllabus_content}
    """
    
    power_pack_content = call_gemini_api(prompt)
    
    if power_pack_content and power_pack_content.strip() not in ["", "{}"]:
        if weak_topics:
            insight_weakness_part = f"It includes a special deep-dive on your weakest topics: {', '.join(weak_topics)}."
        else:
            insight_weakness_part = "You've shown strong mastery across all topics, so no deep-dive section was needed. Well done!"
            
        ai_insight = (
            f"This Power Pack is personalized for you. Based on your activity, we've identified you as a '{persona.get('name')}' and tailored the deep-dive section to match. The main notes are a universal cheat-sheet for quick revision. {insight_weakness_part}"
        )
        
        new_pack = PowerPack.objects.create(
            user=user,
            study_plan=study_plan,
            content=power_pack_content,
            ai_insight=ai_insight
        )
        return new_pack
        
    return None
# In learning/ai_utils.py, add this new function
# In learning/ai_utils.py
# In learning/ai_utils.py, replace this entire function
import json
from django.db import transaction

def generate_completion_quiz(user, lesson):
    """
    Generates a short, 5-question quiz specifically for a single lesson.
    Uses lesson.title if lesson.content is empty. Output is expected from Gemini as JSON.
    """
    # Construct prompt
    prompt = f"""
    You are an expert quiz creator. Your task is to generate a short 5-question quiz about the topic: "{lesson.title}".
    Use the provided lesson content below if it's available, otherwise, use your own knowledge on the topic.
    The quiz must include at least one "multiple-choice" and one "open-ended" question.

    The entire output MUST be a single, valid JSON object with a key "questions".
    Each question object must have: "question_type", "category", "question_text", "answers" (for MC), and "grading_rubric" (for OE).

    --- REFERENCE LESSON CONTENT (may be empty) ---
    {lesson.content}

    IMPORTANT: Your entire response must be a single, valid JSON object. Do not include any other text.

    For a "multiple-choice" question, the structure must be:
    {{
      "topic": "...",
      "question_type": "multiple-choice",
      "question_text": "...",
      "answers": [ "...", "...", "...", "..." ],
      "correct_answer": <index>,
      "category": "..."
    }}

    For an "open-ended" question, the structure must be:
    {{
      "topic": "...",
      "question_type": "open-ended",
      "question_text": "...",
      "grading_rubric": "A brief markdown rubric explaining what a correct answer should contain.",
      "category": "..."
    }}
    """

    try:
        # Call Gemini API
        response_text = call_gemini_api(prompt)

        if not response_text or response_text.strip() == '{}':
            raise ValueError("API returned an empty response.")

        # Clean and parse the JSON
        cleaned_text = response_text.replace('```json', '').replace('```', '').strip()

        try:
            quiz_data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse JSON from Gemini response. Raw response: " + cleaned_text)

        if not isinstance(quiz_data, dict):
            raise ValueError("Parsed quiz_data is not a dictionary.")

        # Create quiz and questions in the database
        with transaction.atomic():
            new_quiz = Quiz.objects.create(
                user=user,
                lesson=lesson,
                title=f"Completion Quiz for: {lesson.title}",
                quiz_type=Quiz.LESSON_QUIZ,
                ai_insight="Pass this quiz to complete the lesson and update your mastery score."
            )

            for q_data in quiz_data.get('questions', []):
                question = Question.objects.create(
                    quiz=new_quiz,
                    lesson=lesson,
                    text=q_data.get('question_text'),
                    question_type=q_data.get('question_type'),
                    category=q_data.get('category', 'CONCEPTUAL'),
                    grading_rubric=q_data.get('grading_rubric')
                )

                if q_data.get('question_type') == 'multiple-choice':
                    answers = q_data.get('answers', [])
                    correct_index = q_data.get('correct_answer')

                    for i, ans_text in enumerate(answers):
                        Answer.objects.create(
                            question=question,
                            text=ans_text,
                            is_correct=(i == correct_index)
                        )

        return new_quiz

    except Exception as e:
        print(f"Error in generate_completion_quiz for lesson '{lesson.title}': {e}")
        return None


# In learning/ai_utils.py
# In learning/ai_utils.py, REPLACE this function
def generate_targeted_flashcards_for_review(user, lesson, flashcard_set=None):
    """
    Analyzes a lesson a user is weak in and generates 3-5 highly targeted
    Q&A flashcards to help them review the core concepts.
    Can optionally add the new cards to a specified FlashcardSet.
    """
    if not lesson.content or not lesson.content.strip():
        return

    prompt = f"""
    A student has a low mastery score in the lesson titled "{lesson.title}".
    Analyze the following lesson content and generate a list of 5-7 points to remember in flashcards format
    that target the most important concepts.

    The response MUST be a single, valid JSON object with a "flashcards" key.
    The value should be a list of objects, where each object has a "question" and "answer" key.
    --- LESSON CONTENT ---
    {lesson.content}
    """
    
    response_text = call_gemini_api(prompt)
    try:
        data = json.loads(response_text)
        flashcards_data = data.get('flashcards', [])

        for card_data in flashcards_data:
            # Create the flashcard. We set created_by to null because the AI wrote it.
            flashcard = Flashcard.objects.create(
                lesson=lesson,
                question=card_data['question'],
                answer=card_data['answer'],
                flashcard_set=flashcard_set, # Link to the provided set
                created_by=None
            )
            UserFlashcard.objects.get_or_create(user=user, flashcard=flashcard)

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error generating targeted flashcards for {lesson.title}: {e}")


# In learning/ai_utils.py

# =============================================================================
# REVISION CARD SYSTEM
# =============================================================================

def generate_revision_card_content(lesson):
    """
    Generates a concise, point-wise summary of a lesson's content,
    perfect for a quick revision flashcard.
    """
    if not lesson.content or not lesson.content.strip():
        # Use the title as a fallback if content is empty
        content_for_prompt = f"The topic is '{lesson.title}'."
    else:
        content_for_prompt = lesson.content

    prompt = f"""
    You are an academic assistant. Your task is to create a concise "Revision Card" for the following topic.
    The content should be a very brief, point-wise summary focusing on the absolute key information.
    Use markdown for formatting, especially headings, bullet points, and bolding for key terms.
    This is for a user who is revising, so keep it clear and to the point.

    --- REFERENCE CONTENT ---
    {content_for_prompt}
    """
    return call_gemini_api(prompt)


def get_daily_revision_deck(user):
    """
    Constructs a personalized daily deck of Revision Cards for a user.

    The deck includes:
    1.  Spaced Repetition Queue: Cards due for review today based on the schedule.
    2.  Upcoming Cards: Cards scheduled for review in the next 7 days.
    """
    today = timezone.now().date()
    
    # Get cards due for review today or in the past
    due_cards = user.revision_cards.filter(
        next_review_date__lte=today
    ).select_related('lesson').order_by('next_review_date')

    # Get cards scheduled for the upcoming week
    upcoming_cards = user.revision_cards.filter(
        next_review_date__gt=today,
        next_review_date__lte=today + timedelta(days=7)
    ).select_related('lesson').order_by('next_review_date')

    return {
        'due_cards': list(due_cards),
        'upcoming_cards': list(upcoming_cards),
    }


# In learning/ai_utils.py, add this new function

def generate_holistic_ai_summary(user):
    """
    Generates a comprehensive, AI-powered summary of a user's learning habits,
    strengths, and areas for improvement for their main profile page.
    """
    # 1. Gather all raw data points
    avg_score = QuizAttempt.objects.filter(user=user).aggregate(avg=Avg('score'))['avg'] or 0
    total_completed = UserLessonProgress.objects.filter(user=user).count()
    total_tasks = ScheduledTask.objects.filter(scheduled_day__study_plan__user=user).count()
    missed_tasks = ScheduledTask.objects.filter(scheduled_day__study_plan__user=user, completed=False).count()
    
    # Calculate study consistency from heatmap data
    heatmap_data = get_heatmap_data(user)
    study_days = len(heatmap_data)
    consistency = (study_days / 90) * 100 if study_days > 0 else 0

    # Get strengths and weaknesses
    mastery_profile = get_user_analytics_profile(user)
    strengths = [s.lesson.title for s in mastery_profile.get('strongest_topics', [])]
    weaknesses = [w.lesson.title for w in mastery_profile.get('weakest_topics', [])]
    
    # 2. Build the detailed prompt for the AI
    prompt = f"""
    You are an expert learning coach AI. Analyze the following data profile for a student and generate a set of encouraging and actionable insights.

    --- USER DATA PROFILE ---
    - Average Quiz Score: {avg_score:.0f}%
    - Total Lessons Completed: {total_completed}
    - Schedule Adherence: {((total_tasks - missed_tasks) / total_tasks) * 100 if total_tasks > 0 else 100:.0f}% ({missed_tasks} missed tasks)
    - Study Consistency (last 90 days): {consistency:.0f}%
    - Strongest Topics: {', '.join(strengths) if strengths else 'None identified'}
    - Weakest Topics: {', '.join(weaknesses) if weaknesses else 'None identified'}
    - Learning Persona: {mastery_profile['learning_persona']['name']} ({mastery_profile['learning_persona']['description']})

    --- YOUR TASK ---
    Return a single, valid JSON object with three keys: "overall_summary", "actionable_advice", and "learning_style_insight".
    - "overall_summary": A brief, encouraging paragraph summarizing the user's habits and progress.
    - "actionable_advice": A bulleted list (as a single string with '\\n- ') with 2-3 specific, positive suggestions for what to do next.
    - "learning_style_insight": A short paragraph explaining how their learning persona might be affecting their progress and how they can leverage it.

    Keep the tone positive, insightful, and motivational.
    """

    # 3. Call the API and return the structured data
    response_text = call_gemini_api(prompt)
    try:
        # Clean up potential markdown formatting from the response
        cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        return {
            "overall_summary": "Could not generate an AI summary at this time. Keep up the great work!",
            "actionable_advice": "- Try completing a new lesson today.\n- Review one of your weaker topics in the Revision Card Center.",
            "learning_style_insight": "The more you interact with different content, the better I can tailor recommendations to your style."
        }