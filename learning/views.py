# learning/views.py

"""
Django Views for Learning Management System

This module contains all the view functions for the learning app,
organized into logical sections for better maintainability.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard Library Imports
import json
import random
import time
import re
from itertools import count
from datetime import date, datetime, timedelta

# Third-party Imports
import markdown2

# Django Imports
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from django.http import JsonResponse
from django.db.models import Count, Avg

# Local Model Imports
from .models import (
    Answer, Flashcard, FlashcardSet, KnowledgeGap, PowerPack, QuizAttempt, StudyPlan, Module, Lesson, UserAnswer, UserLessonProgress, Badge, Mastery, 
    UserFlashcard, Quiz, Question, ScheduledDay, ScheduledTask,RevisionCard,
    UserInteraction,
)

# Local AI Utility Imports
from .ai_utils import (
    detect_learning_persona, find_knowledge_gaps, find_weakness_of_the_week, generate_pre_exam_power_pack, generate_quiz_questions, generate_syllabus_structure,
    find_youtube_video, generate_lesson_details, get_content_recommendation, get_generative_ai_explanation,
    get_heatmap_data, perform_daily_schedule_maintenance, recommend_next_lesson, get_job_market_skills, update_mastery,
    get_mindset_coach_message, find_study_pals, generate_flashcards_for_lesson,
    update_flashcard_review, generate_personalized_project, create_study_schedule,
    create_advanced_schedule, get_single_content_piece,get_single_content_piece, get_daily_flashcard_review_deck, # <-- ADD this new import
    update_flashcard_review, 
    generate_automated_quiz, generate_revision_card_content, # <-- ADD this
    get_daily_revision_deck,       # <-- ADD this
    update_mastery 
)
from learning import ai_utils


# =============================================================================
# MAIN DASHBOARD & HOME VIEWS
# =============================================================================

def home_view(request):
    """
    Landing page view - shows dashboard for authenticated users,
    public home for anonymous users.
    """
    if request.user.is_authenticated:
        try:
            user_badges = request.user.userprofile.badges.all()
        except:
            user_badges = []
        
        recommended_lesson = recommend_next_lesson(request.user)
        job_skills = get_job_market_skills()
        user_masteries = Mastery.objects.filter(user=request.user)
        user_mastery_dict = {m.concept_key: m.mastery_score for m in user_masteries}
        
        context = {
            'badges': user_badges,
            'recommended_lesson': recommended_lesson,
            'job_skills': job_skills,
            'user_mastery_dict': user_mastery_dict,
        }
        return render(request, 'learning/dashboard.html', context)
    else:
        return render(request, 'learning/home_public.html')

# In learning/views.py, replace the entire dashboard_view function

# In learning/views.py, replace the entire dashboard_view function

@login_required
def dashboard_view(request):
    """
    Main dashboard showing tasks, schedule, and ALL AI notifications for testing.
    """
    today = timezone.localdate()
    todays_tasks = []
    scheduled_days_info = []
    heatmap_data = get_heatmap_data(request.user)
    latest_plan = StudyPlan.objects.filter(user=request.user).order_by('-created_at').first()

    if latest_plan:
        # Your existing logic for fetching tasks
        all_scheduled_days = ScheduledDay.objects.filter(study_plan=latest_plan).order_by("date")
        for d in all_scheduled_days:
            tasks = d.scheduledtask_set.all()
            scheduled_days_info.append({
                "date": str(d.date), "tasks": [{"lesson_id": t.lesson.id if t.lesson else None, "title": t.lesson.title if t.lesson else t.task_description} for t in tasks]
            })
        scheduled_day_for_today = next((d for d in all_scheduled_days if str(d.date) == str(today)), None)
        if scheduled_day_for_today:
            todays_tasks = list(scheduled_day_for_today.scheduledtask_set.all())

    completed_lessons = UserLessonProgress.objects.filter(user=request.user).select_related('lesson').order_by('-completed_at')

    # --- CHANGE 1: Fetch ALL notifications for the user ---
    notifications = request.user.notifications.all().order_by('-created_at')

    context = {
        "today": today,
        "latest_plan": latest_plan,
        "todays_tasks": todays_tasks,
        "scheduled_days_info": scheduled_days_info,
        'heatmap_data': json.dumps(heatmap_data),
        'notifications': notifications,"completed_lessons": completed_lessons,
    }

    # --- CHANGE 2: We no longer mark them as read ---
    # notifications.update(is_read=True) # This line is now removed/commented out

    return render(request, "learning/dashboard.html", context)

# In learning/views.py, replace this entire function
# In learning/views.py, replace this entire function

# In learning/views.py

from .ai_utils import get_user_analytics_profile # Make sure this is imported at the top

# ... (other view functions) ...

@login_required
def my_progress_view(request):
    """
    Analytics dashboard showing the user's learning profile by calling the 
    master analytics engine.
    """
    # Call the single, centralized AI utility to get the full analytics profile
    analytics_profile = get_user_analytics_profile(request.user)
    
    context = {
        'analytics_profile': analytics_profile,
    }
    
    return render(request, 'learning/my_progress.html', context)
# In learning/views.py

@login_required
def profile_view(request):
    """
    User profile page showing personal information, settings, earned badges,
    and a holistic AI-powered summary of their learning journey.
    """
    user_profile = request.user.userprofile
    
    ai_summary_data = ai_utils.generate_holistic_ai_summary(request.user)
    
    # --- THIS IS THE FIX ---
    # Split the string and then filter out any empty strings from the list
    if ai_summary_data and 'actionable_advice' in ai_summary_data:
        advice_list = ai_summary_data['actionable_advice'].split('\\n- ')
        # Create a new list containing only non-empty, stripped items
        ai_summary_data['actionable_advice_list'] = [item.strip() for item in advice_list if item.strip()]
    # --- END OF FIX ---

    user_badges = user_profile.badges.all()

    context = {
        'user_profile': user_profile,
        'ai_summary': ai_summary_data,
        'user_badges': user_badges,
    }
    return render(request, 'learning/profile_page.html', context)


# =============================================================================
# STUDY PLAN GENERATION & MANAGEMENT
# =============================================================================

@login_required
def generate_plan_view(request):
    """
    AI-powered study plan generation from user input.
    """
    structured_syllabus = None
    
    if request.method == 'POST':
        syllabus_text = request.POST.get('syllabus_text')
        
        if syllabus_text:
            json_response = generate_syllabus_structure(syllabus_text)
            
            try:
                # Clean and parse the AI JSON response
                json_match = re.search(r'\{.*\}', json_response, re.DOTALL)
                if not json_match:
                    raise ValueError("No valid JSON object found in AI response.")
                
                cleaned_json = json_match.group()
                structured_syllabus = json.loads(cleaned_json)
                
            except (ValueError, json.JSONDecodeError, IndexError):
                messages.error(request, "The AI returned an invalid format. Please try again.")
    
    context = {'structured_syllabus': structured_syllabus}
    return render(request, 'learning/generate_plan.html', context)


@login_required
def save_plan_view(request):
    """
    Save the generated study plan structure to database.
    """
    if request.method == 'POST':
        module_titles = request.POST.getlist('module_title')
        plan_title = request.POST.get('plan_title', 'My New Study Plan')
        subject = request.POST.get('subject', 'General')
        
        # Create the main study plan
        study_plan = StudyPlan.objects.create(user=request.user, title=plan_title, subject=subject)
        
        # Create modules and lessons
        for i, module_title in enumerate(module_titles):
            if module_title:
                module = Module.objects.create(
                    study_plan=study_plan, 
                    title=module_title, 
                    order=i
                )
                
                lesson_titles = request.POST.getlist(f'module_{i}_lessons')
                for j, lesson_title in enumerate(lesson_titles):
                    if lesson_title:
                        concept_key = f"{lesson_title.lower().replace(' ','_').replace('/','_')}_{study_plan.id}"
                        Lesson.objects.create(
                            module=module, 
                            title=lesson_title, 
                            order=j, 
                            concept_key=concept_key
                        )
        
        messages.success(request, "Your study plan has been saved successfully!")
        return redirect('my_plans')
    
    return redirect('generate_plan')


@login_required
def my_plans_view(request):
    """
    Display all user's study plans.
    """
    study_plans = StudyPlan.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'learning/my_plans.html', {'study_plans': study_plans})


@login_required
def plan_detail_view(request, plan_id):
    """
    Detailed view of a specific study plan.
    """
    plan = get_object_or_404(StudyPlan, id=plan_id, user=request.user)
    return render(request, 'learning/plan_detail.html', {'plan': plan})


# =============================================================================
# LEARNING CONTENT & LESSON VIEWS
# =============================================================================
# In learning/views.py

# In learning/views.py

# In learning/views.py

# In learning/views.py, replace the lesson_detail_view
from .ai_utils import generate_completion_quiz # Add this import at the top
# In learning/views.py, replace this entire function
# In learning/views.py
from .ai_utils import generate_completion_quiz # Add this import
# In learning/views.py, replace this entire function
# In learning/views.py, replace this entire function

# In learning/views.py, replace this entire function

@login_required
def lesson_detail_view(request, lesson_id):
    """
    Main lesson learning interface. Now also handles the "completed" state.
    """
    lesson = get_object_or_404(Lesson, id=lesson_id)
    video_url = find_youtube_video(lesson.title)
    recommendation_data = get_content_recommendation(request.user, lesson)
    
    # --- NEW: Check if the user has already completed this lesson ---
    is_completed = UserLessonProgress.objects.filter(user=request.user, lesson=lesson).exists()

    completion_quiz = None
    quiz_insight = ""

    # Only prepare a quiz and insight if the lesson is NOT completed
    if not is_completed:
        completion_quiz = Quiz.objects.filter(
            user=request.user, 
            lesson=lesson, 
            quiz_type=Quiz.LESSON_QUIZ
        ).first()

        if not completion_quiz:
            completion_quiz = generate_completion_quiz(request.user, lesson)

        try:
            mastery = Mastery.objects.get(user=request.user, lesson=lesson)
            if mastery.mastery_score < 0.6:
                quiz_insight = f"Your last mastery score on '{lesson.title}' was a bit low ({mastery.mastery_score:.0%}). This quiz is a great opportunity to improve it!"
            else:
                quiz_insight = f"You're doing great on '{lesson.title}'! This quiz is a chance to solidify your knowledge and push your mastery even higher."
        except Mastery.DoesNotExist:
            quiz_insight = "This will be your first quiz on this topic. Passing it will establish your initial mastery score."
    
    context = {
        'lesson': lesson,
        'video_url': video_url,
        'recommendation_data': recommendation_data,
        'is_completed': is_completed, # Pass the completion status to the template
        'completion_quiz': completion_quiz,
        'quiz_insight': quiz_insight,
    }
    return render(request, 'learning/lesson_detail.html', context)
@login_required
def get_lesson_content_view(request, lesson_id, content_type):
    """
    AJAX endpoint to fetch lesson content dynamically.
    Returns formatted HTML content for different content types.
    """
    lesson = get_object_or_404(Lesson, id=lesson_id)
    raw_content = get_single_content_piece(lesson.title, content_type)
    html_content = markdown2.markdown(raw_content, extras=['fenced-code-blocks'])
    
    return JsonResponse({'html_content': html_content})


@login_required
def log_interaction_view(request):
    """
    AJAX endpoint to log user interactions for analytics.
    """
    if request.method == 'POST':
        data = json.loads(request.body)
        lesson_id = data.get('lesson_id')
        interaction_type = data.get('interaction_type')
        feedback = data.get('feedback', None)

        lesson = get_object_or_404(Lesson, id=lesson_id)

        UserInteraction.objects.create(
            user=request.user,
            lesson=lesson,
            interaction_type=interaction_type,
            feedback=feedback
        )
        return JsonResponse({'status': 'ok'})
    
    return JsonResponse({'status': 'error'}, status=400)


@login_required
def ask_ai_view(request, lesson_id):
    """
    AI tutor interface for getting explanations and help.
    """
    lesson = get_object_or_404(Lesson, id=lesson_id)
    mode = request.POST.get('mode', 'analogy')
    custom_doubt = request.POST.get('custom_doubt', '')

    raw_explanation = get_generative_ai_explanation(request.user, lesson, mode, custom_doubt)
    
    # Process special markers for Mermaid diagrams
    processed_explanation = raw_explanation.replace(
        '[MERMAID_START]', '<div class="mermaid">'
    ).replace('[MERMAID_END]', '</div>')
    
    html_explanation = markdown2.markdown(processed_explanation, extras=['fenced-code-blocks'])
    
    context = {
        'lesson': lesson, 
        'ai_explanation': html_explanation
    }
    return render(request, 'learning/lesson_detail.html', context)


@login_required
def mark_lesson_complete_view(request, lesson_id):
    """
    Mark a lesson as completed and handle badge rewards.
    """
    if request.method == 'POST':
        lesson = get_object_or_404(Lesson, id=lesson_id)
        progress, created = UserLessonProgress.objects.get_or_create(
            user=request.user, 
            lesson=lesson
        )
        
        # Generate flashcards for new completions
        if created:
            generate_flashcards_for_lesson(request.user, lesson)
        
        # Award first lesson badge
        if UserLessonProgress.objects.filter(user=request.user).count() == 1:
            try:
                first_badge = Badge.objects.get(name="First Lesson Complete")
                request.user.userprofile.badges.add(first_badge)
                messages.success(
                    request, 
                    f"Congratulations! You've earned the '{first_badge.name}' badge!"
                )
            except Badge.DoesNotExist:
                pass

        messages.success(request, f"Lesson '{lesson.title}' marked as complete!")
        return redirect('plan_detail', plan_id=lesson.module.study_plan.id)
    
    return redirect('lesson_detail', lesson_id=lesson_id)


# =============================================================================
# QUIZ & ASSESSMENT VIEWS
# =============================================================================

@login_required
def quiz_view(request, lesson_id):
    """
    Quiz interface for lesson assessment.
    """
    lesson = get_object_or_404(Lesson, id=lesson_id)
    return render(request, 'learning/quiz_page.html', {'lesson': lesson})

# learning/views.py
# learning/views.py
# In learning/views.py
from collections import defaultdict
from .ai_utils import analyze_quiz_and_reschedule, update_mastery # Ensure all are imported

# ... (other views)

# In learning/views.py
from collections import defaultdict
from .ai_utils import analyze_quiz_and_reschedule, update_mastery, grade_open_ended_answer # Add grade_open_ended_answer
# In learning/views.py
from collections import defaultdict
from .ai_utils import analyze_quiz_and_reschedule, update_mastery, grade_open_ended_answer
# In learning/views.py, replace the entire submit_quiz_view function

# In learning/views.py, replace the entire submit_quiz_view function

# In learning/views.py
from collections import defaultdict # Ensure this is at the top of your file
from .models import Quiz, QuizAttempt, UserAnswer, Answer, UserLessonProgress # etc.
from .ai_utils import grade_open_ended_answer, update_mastery, analyze_quiz_and_reschedule # etc.

# ... (other views) ...

@login_required
def submit_quiz_view(request, quiz_id):
    if request.method == 'POST':
        quiz = get_object_or_404(Quiz, id=quiz_id)
        questions = quiz.questions.all()
        total_questions = questions.count()

        attempt = QuizAttempt.objects.create(
            quiz=quiz, user=request.user, score=0, total_questions=total_questions
        )
        
        lesson_scores = defaultdict(lambda: {'score_sum': 0.0, 'count': 0})

        for question in questions:
            user_input = request.POST.get(f'question_{question.id}')
            if user_input:
                if question.question_type == 'open-ended':
                    grading_result = grade_open_ended_answer(question.text, question.grading_rubric, user_input)
                    ai_score = grading_result.get('score', 0.0)
                    is_correct_from_ai = (ai_score >= 0.7)
                    UserAnswer.objects.create(
                        quiz_attempt=attempt, question=question, open_ended_response=user_input,
                        ai_feedback=grading_result.get('feedback'), ai_score=ai_score, is_correct=is_correct_from_ai
                    )
                    if question.lesson:
                        lesson_scores[question.lesson]['score_sum'] += ai_score
                        lesson_scores[question.lesson]['count'] += 1
                
                else: # Multiple-Choice
                    selected_answer = get_object_or_404(Answer, id=user_input)
                    is_correct = selected_answer.is_correct
                    UserAnswer.objects.create(
                        quiz_attempt=attempt, question=question,
                        selected_answer=selected_answer, is_correct=is_correct
                    )
                    if question.lesson:
                        lesson_scores[question.lesson]['score_sum'] += 1 if is_correct else 0
                        lesson_scores[question.lesson]['count'] += 1
        
        total_score_sum = sum(data['score_sum'] for data in lesson_scores.values())
        final_score_percent = (total_score_sum / total_questions) * 100 if total_questions > 0 else 0
        
        attempt.score = int(total_score_sum)
        attempt.save()

        insights = []
        
        if quiz.quiz_type == Quiz.LESSON_QUIZ and quiz.lesson:
            PASSING_SCORE = 60
            
            # --- BUG FIX STARTS HERE ---
            # 1. The `update_mastery` call is now OUTSIDE the 'if' block.
            #    This ensures it runs for EVERY lesson quiz, pass or fail.
            update_mastery(request.user, quiz.lesson, final_score_percent / 100.0)

            if final_score_percent >= PASSING_SCORE:
                # 2. If they passed, we still create the completion record as before.
                UserLessonProgress.objects.get_or_create(
                    user=request.user,
                    lesson=quiz.lesson,
                    defaults={'quiz_score': final_score_percent}
                )
                insights.append(f"Congratulations! You passed with {final_score_percent:.0f}% and have completed the lesson '{quiz.lesson.title}'.")
            else:
                # 3. The failure message is updated to confirm the mastery score was saved.
                insights.append(f"Not quite. You scored {final_score_percent:.0f}%, but you need {PASSING_SCORE}% to pass. Your new mastery score has been recorded. Please review the material and try again!")
            # --- BUG FIX ENDS HERE ---

        else:
            # This part for other quiz types remains unchanged
            for lesson, scores in lesson_scores.items():
                average_score = (scores['score_sum'] / scores['count']) if scores['count'] > 0 else 0.0
                insight = update_mastery(request.user, lesson, average_score)
                if insight:
                    insights.append(insight)
            analyze_quiz_and_reschedule(attempt)

        request.session['quiz_insights'] = insights
        
        return redirect('quiz_results', attempt_id=attempt.id)
    
    return redirect('dashboard')
# In learning/views.py

# ... (place this near your other quiz views) ...
@login_required
def quiz_results_view(request, attempt_id):
    """
    Displays the results summary page after a quiz is submitted.
    """
    attempt = get_object_or_404(QuizAttempt, id=attempt_id, user=request.user)
    context = {'attempt': attempt}
    return render(request, 'learning/quiz_results.html', context)
# =============================================================================
# SPACED REPETITION & REVIEW SYSTEM
# =============================================================================
# In learning/views.py, in the SPACED REPETITION & REVIEW SYSTEM section
# In learning/views.py, REPLACE the flashcard_dashboard_view
from .ai_utils import generate_targeted_flashcards_for_review # Ensure this is imported

@login_required
def flashcard_dashboard_view(request):
    """
    Main dashboard for the flashcard center. Displays AI-powered review decks,
    user-created sets, and suggestions. Now handles auto-creation of suggested sets.
    """
    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'create_suggested_set':
            lesson_id = request.POST.get('lesson_id')
            if lesson_id:
                lesson = get_object_or_404(Lesson, id=lesson_id, module__study_plan__user=request.user)
                
                # 1. Create the new flashcard set
                new_set = FlashcardSet.objects.create(
                    user=request.user,
                    title=f"AI Review: {lesson.title}",
                    description=f"AI-generated flashcards to review '{lesson.title}'."
                )
                # 2. Generate flashcards and link them to this new set
                generate_targeted_flashcards_for_review(request.user, lesson, flashcard_set=new_set)
                
                messages.success(request, f"Successfully created and populated the set '{new_set.title}'.")
                return redirect('flashcard_set_view', set_id=new_set.id)
        else:
            # Logic for creating a new user-defined flashcard set from the modal
            set_title = request.POST.get('set_title')
            set_description = request.POST.get('set_description')
            if set_title:
                new_set = FlashcardSet.objects.create(
                    user=request.user, title=set_title, description=set_description
                )
                messages.success(request, f"Successfully created flashcard set '{new_set.title}'.")
                return redirect('flashcard_set_view', set_id=new_set.id)

    # Fetch AI-powered review decks (now includes upcoming cards)
    review_decks = get_daily_flashcard_review_deck(request.user)
    
    # Fetch all user-created flashcard sets
    user_sets = FlashcardSet.objects.filter(user=request.user).order_by('-created_at')

    # Fetch topics with mastery < 80% to suggest for new card creation
    suggestion_topics = Mastery.objects.filter(
        user=request.user, mastery_score__lt=0.8, lesson__isnull=False
    ).select_related('lesson').order_by('mastery_score')

    context = {
        'srs_cards': review_decks['srs_cards'],
        'todays_topic_cards': review_decks['todays_topic_cards'],
        'upcoming_cards': review_decks['upcoming_cards'], # Pass new data to template
        'user_sets': user_sets,
        'suggestion_topics': suggestion_topics,
    }
    return render(request, 'learning/flashcard_dashboard.html', context)

@login_required
def flashcard_set_view(request, set_id):
    """
    View for managing and reviewing a specific user-created flashcard set.
    """
    flashcard_set = get_object_or_404(FlashcardSet, id=set_id, user=request.user)

    if request.method == 'POST':
        # Logic for adding a new flashcard to this set
        question_text = request.POST.get('question')
        answer_text = request.POST.get('answer')
        if question_text and answer_text:
            new_card = Flashcard.objects.create(
                flashcard_set=flashcard_set,
                created_by=request.user,
                question=question_text,
                answer=answer_text
            )
            # Link this new card to the user for spaced repetition tracking
            UserFlashcard.objects.create(user=request.user, flashcard=new_card)
            messages.success(request, "New flashcard added to your set.")
            return redirect('flashcard_set_view', set_id=set_id)

    # Get all UserFlashcard objects associated with this set for review
    cards_in_set = UserFlashcard.objects.filter(
        user=request.user,
        flashcard__flashcard_set=flashcard_set
    ).select_related('flashcard')
    
    context = {
        'flashcard_set': flashcard_set,
        'cards_in_set': cards_in_set
    }
    return render(request, 'learning/flashcard_set_view.html', context)

@login_required
def flashcard_review_session_view(request):
    """
    The main interface for reviewing a deck of flashcards.
    """
    card_ids_str = request.GET.get('cards', '')
    if not card_ids_str:
        messages.error(request, "No cards selected for review.")
        return redirect('flashcard_dashboard')

    card_ids = [int(id) for id in card_ids_str.split(',')]
    
    # Fetch the specific UserFlashcard objects for this session
    review_cards = UserFlashcard.objects.filter(id__in=card_ids, user=request.user).select_related('flashcard')

    context = {
        'review_cards': review_cards,
        'deck_title': request.GET.get('title', 'Review Session') # Get title from URL parameter
    }
    return render(request, 'learning/flashcard_review_session.html', context)

from .ai_utils import generate_targeted_review_notes
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
import markdown2

@login_required
def review_session_view(request, task_id):
    # Ensure the task exists and belongs to the current user
    task = get_object_or_404(ScheduledTask, id=task_id, scheduled_day__study_plan__user=request.user)

    # *** This is the core of the fix ***
    # Check if the task has a linked triggering_attempt before trying to access it.
    if task.triggering_attempt:
        # If it exists, proceed as normal
        lesson_title = task.triggering_attempt.quiz.title.replace("Practice Test on ", "")
    else:
        # Handle the case where there is no triggering_attempt
        # You'll need to decide on a reasonable fallback value here.
        # For example, you could get the lesson title from another source or set a default.
        # Let's assume you have a title field on the task itself or want a generic title.
        lesson_title = "Unknown Lesson"  # Fallback title

    # Extract the weak category and topic from the task description
    description = task.task_description
    weak_category = "Conceptual" # Default
    if "Conceptual" in description:
        weak_category = "Conceptual"
    elif "Code" in description:
        weak_category = "Code"
    elif "Terminology" in description:
        weak_category = "Terminology"

    # Call the AI to generate targeted notes
    raw_notes = generate_targeted_review_notes(lesson_title, weak_category)
    review_notes = markdown2.markdown(raw_notes, extras=['fenced-code-blocks'])

    context = {
        'task': task,
        'review_notes': review_notes
    }
    return render(request, 'learning/review_session.html', context)


@login_required
def process_review_answer_view(request, user_flashcard_id):
    """
    Process flashcard review answer and update spaced repetition schedule.
    """
    if request.method == 'POST':
        user_flashcard = get_object_or_404(
            UserFlashcard, 
            id=user_flashcard_id, 
            user=request.user
        )
        recalled_correctly = request.POST.get('recalled_correctly') == 'yes'
        update_flashcard_review(user_flashcard, recalled_correctly)
    
    return redirect('review_session')


# =============================================================================
# SCHEDULING & PLANNING VIEWS
# =============================================================================

@login_required
def scheduler_view(request):
    """
    AI-powered study schedule generation based on exam dates and selected modules.
    """
    study_plans = StudyPlan.objects.filter(user=request.user)
    
    if request.method == 'POST':
        plan_id = request.POST.get('study_plan')
        exam_date_str = request.POST.get('exam_date')
        backup_days = request.POST.get('backup_days', 0)
        selected_module_ids = request.POST.getlist('modules')
        
        if plan_id and exam_date_str and selected_module_ids:
            study_plan = get_object_or_404(StudyPlan, id=plan_id, user=request.user)
            exam_date = datetime.strptime(exam_date_str, '%Y-%m-%d').date()
            
            # Update plan with exam date
            study_plan.exam_date = exam_date
            study_plan.save()

            # Generate AI schedule
            schedule_data = create_advanced_schedule(
                study_plan, 
                exam_date, 
                selected_module_ids, 
                backup_days
            )
            
            # Clear existing schedule
            ScheduledDay.objects.filter(study_plan=study_plan).delete()
            
            # Save new schedule to database
            for day_data in schedule_data:
                day, created = ScheduledDay.objects.get_or_create(
                    study_plan=study_plan, 
                    date=day_data['date']
                )
                ScheduledTask.objects.create(
                    scheduled_day=day,
                    lesson=day_data.get('lesson'), 
                    task_description=day_data['description'],
                    task_type=day_data['task_type']
                )
            
            messages.success(request, "Your new schedule has been generated and saved!")
            return redirect('my_plans')

    context = {'study_plans': study_plans}
    return render(request, 'learning/scheduler.html', context)


# =============================================================================
# COMMUNITY & ADDITIONAL FEATURES
# =============================================================================

@login_required
def project_generation_view(request):
    """
    AI-generated personalized project suggestions.
    """
    project_idea = None
    if request.method == 'POST':
        project_idea = generate_personalized_project(request.user)
    
    return render(request, 'learning/project_page.html', {'project_idea': project_idea})


@login_required
def studypals_view(request):
    """
    Find study partners and mentors based on learning progress.
    """
    suggested_peers, suggested_mentors, weak_concept_key = find_study_pals(request.user)
    weak_lesson = Lesson.objects.filter(concept_key=weak_concept_key).first()
    
    context = {
        'suggested_peers': suggested_peers,
        'suggested_mentors': suggested_mentors,
        'weak_lesson': weak_lesson
    }
    return render(request, 'learning/studypals_page.html', context)


# =============================================================================
# LEGACY/UTILITY VIEWS
# =============================================================================

@login_required
def module_list_view(request):
    """
    Browse global modules (legacy view - less relevant now).
    Note: This is now less relevant as content is tied to specific StudyPlans.
    You might adapt this later to browse public or template plans.
    """
    modules = Module.objects.filter(study_plan__isnull=True).order_by('order')
    completed_lessons = UserLessonProgress.objects.filter(
        user=request.user
    ).values_list('lesson_id', flat=True)
    
    context = {
        'modules': modules,
        'completed_lessons': completed_lessons,
    }
    return render(request, 'learning/module_list.html', context)

#phase 4 

# learning/views.py
@login_required
def generate_quiz_view(request, task_id):
    task = get_object_or_404(ScheduledTask, id=task_id, scheduled_day__study_plan__user=request.user)
    topics = [task.lesson.title] if task.lesson else [task.task_description]

    if request.method == 'POST':
        num_questions = request.POST.get('num_questions', 5)
        question_type = request.POST.get('question_type', 'multiple-choice')
        
        quiz_json_str = generate_quiz_questions(topics, num_questions, question_type)
        try:
            quiz_data = json.loads(quiz_json_str)
        except json.JSONDecodeError:
            messages.error(request, "The AI failed to generate a quiz. Please try again.")
            return redirect('dashboard')

        context = { 'task': task, 'quiz_data': quiz_data }
        return render(request, 'learning/generated_quiz.html', context)

    # If it's a GET request, show the setup form
    context = { 'task': task }
    return render(request, 'learning/quiz_setup.html', context)
# learning/views.py
# learning/views.py
# In learning/views.py
# In learning/views.py
from collections import Counter # Make sure Counter is imported
from django.contrib import messages # Make sure messages is imported

# ... (other views) ...
# In learning/views.py
# In learning/views.py
from collections import Counter
from django.contrib import messages
from django.db.models import Avg

# ... (other imports)

# In learning/views.py, replace the entire practice_center_view function

@login_required
def practice_center_view(request):
    study_plans = StudyPlan.objects.filter(user=request.user).prefetch_related('module_set__lesson_set')

    if request.method == 'POST':
        lesson_ids = request.POST.getlist('lessons')
        num_questions = request.POST.get('num_questions', 5)
        question_type = request.POST.get('question_type', 'multiple-choice')
        difficulty = request.POST.get('difficulty', 'Medium')

        if not lesson_ids:
            messages.error(request, "Please select at least one topic.")
            return redirect('practice_center')

        lessons = Lesson.objects.filter(id__in=lesson_ids)
        lesson_titles = [lesson.title for lesson in lessons]
        
        difficulty_profile = {'difficulty': difficulty}
        debug_message = f"Generating a '{difficulty}' quiz of type '{question_type}'."

        if difficulty == 'Adaptive':
            # --- THIS IS THE CORRECTED LINE ---
            # We now filter by the 'lesson' object itself, not the old 'concept_key'.
            mastery_records = Mastery.objects.filter(user=request.user, lesson__in=lessons)
            # --- END OF CORRECTION ---
            
            avg_mastery = mastery_records.aggregate(avg=Avg('mastery_score'))['avg'] or 0.0
            difficulty_profile['mastery_score'] = f"{avg_mastery:.0%}"

            if avg_mastery < 0.4:
                difficulty_profile['difficulty'] = 'Easy'
            elif avg_mastery < 0.75:
                difficulty_profile['difficulty'] = 'Medium'
            else:
                difficulty_profile['difficulty'] = 'Hard'

            incorrect_answers = UserAnswer.objects.filter(
                quiz_attempt__user=request.user, 
                question__lesson__in=lessons, 
                is_correct=False
            ).values_list('question__category', flat=True)
            
            weaknesses = dict(Counter(incorrect_answers))
            difficulty_profile['weaknesses'] = weaknesses
            
            debug_message = (
                f"AI PROFILE [Adaptive Mode]: Difficulty set to '{difficulty_profile['difficulty']}' "
                f"based on your average mastery of {difficulty_profile['mastery_score']}. "
            )
            if weaknesses:
                weakness_str = ", ".join([f"{cat} ({num} wrong)" for cat, num in weaknesses.items()])
                debug_message += f"Targeting weaknesses in: {weakness_str}."
            else:
                debug_message += "No specific weaknesses found for these topics."
        
        request.session['ai_insight_message'] = debug_message
        
        lesson_map = {lesson.title: lesson for lesson in lessons}
        quiz_json_str = generate_quiz_questions(lesson_titles, num_questions, question_type, difficulty_profile)
        
        try:
            json_match = re.search(r'\{.*\}', quiz_json_str, re.DOTALL)
            if not json_match: raise ValueError("No valid JSON found in AI response.")
            cleaned_json = json_match.group()
            quiz_data = json.loads(cleaned_json)
            
            linked_lesson = lessons.first() if lessons.count() == 1 else None
            
            quiz_title = f"Practice Test on {', '.join(lesson_titles)}"
            # Your Quiz model no longer has `is_ai_generated`, so we set the type.
            new_quiz = Quiz.objects.create(
                user=request.user, title=quiz_title, quiz_type='lesson', lesson=linked_lesson
            )
            
            for q_data in quiz_data['questions']:
                question_topic_title = q_data.get('topic')
                question_lesson = lesson_map.get(question_topic_title)
                category = q_data.get('category', 'CONCEPTUAL').upper()
                question = Question.objects.create(
                    quiz=new_quiz, lesson=question_lesson, text=q_data['question_text'], category=category,
                    question_type=q_data.get('question_type', 'multiple-choice'),
                    grading_rubric=q_data.get('grading_rubric')
                )
                if q_data.get('question_type') == 'multiple-choice':
                    for i, ans_text in enumerate(q_data['answers']):
                        Answer.objects.create(
                            question=question, text=ans_text, is_correct=(i == q_data['correct_answer'])
                        )
            return redirect('take_quiz', quiz_id=new_quiz.id)

        except (ValueError, json.JSONDecodeError, KeyError) as e:
            messages.error(request, f"The AI failed to generate a valid quiz ({e}). Please try again.")
            return redirect('practice_center')
    
    context = { 'study_plans': study_plans }
    return render(request, 'learning/practice_center.html', context)

# In learning/views.py
# learning/views.py
# ... (all your other imports should be at the top) ...

@login_required
def take_quiz_view(request, quiz_id):
    """
    Renders the quiz interface for a user to take a quiz.
    """
    quiz = get_object_or_404(Quiz, id=quiz_id, user=request.user)
    
    # Get the AI insight message from the session and remove it
    ai_insight = request.session.pop('ai_insight_message', None)
    
    context = {
        'quiz': quiz,
        'ai_insight': ai_insight, # Pass the message to the template
    }
    return render(request, 'learning/take_quiz.html', context)

@login_required
def quiz_history_view(request):
    """
    Displays all past quiz attempts with a correctly calculated overall score.
    """
    attempts = QuizAttempt.objects.filter(user=request.user).order_by('-timestamp').prefetch_related('useranswer_set')

    # Calculate the true score for each attempt
    attempts_with_scores = []
    for attempt in attempts:
        total_score = 0
        num_answers = attempt.useranswer_set.count()

        for answer in attempt.useranswer_set.all():
            if answer.question.question_type == 'open-ended':
                total_score += answer.ai_score or 0.0
            else:
                if answer.is_correct:
                    total_score += 1
        
        # Calculate overall percentage for display
        overall_percent = int((total_score / num_answers) * 100) if num_answers > 0 else 0
        
        attempts_with_scores.append({
            'attempt': attempt,
            'overall_percent': overall_percent,
        })

    context = { 'attempts_with_scores': attempts_with_scores }
    return render(request, 'learning/quiz_history.html', context)

@login_required
def quiz_review_view(request, attempt_id):
    attempt = get_object_or_404(QuizAttempt, id=attempt_id, user=request.user)
    # Get all the user's answers for this attempt
    user_answers = UserAnswer.objects.filter(quiz_attempt=attempt)

    context = {
        'attempt': attempt,
        'user_answers': user_answers,
    }
    return render(request, 'learning/quiz_review.html', context)

# learning/views.py
from .ai_utils import get_tutor_response # Add this import
# learning/views.py
@login_required
def ask_tutor_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        lesson_id = data.get('lesson_id')
        question_text = data.get('question_text')
        
        lesson = get_object_or_404(Lesson, id=lesson_id)
        
        # FIXED: Generate content if lesson.content is empty
        lesson_content = lesson.content
        if not lesson_content or not lesson_content.strip():
            # Generate notes as the lesson content
            lesson_content = get_single_content_piece(lesson.title, 'notes')
            
            # Optionally save it to the database for future use
            lesson.content = lesson_content
            lesson.save()
        
        raw_response = get_tutor_response(lesson_content, question_text)
        html_response = markdown2.markdown(raw_response, extras=['fenced-code-blocks'])
        
        # Log the interaction
        UserInteraction.objects.create(
            user=request.user,
            lesson=lesson,
            interaction_type='asked_tutor'
        )
        
        return JsonResponse({'response': html_response})

    return JsonResponse({'error': 'Invalid request'}, status=400)

from .models import ScheduledTask # Make sure ScheduledTask is imported

# ... (other imports)
# In learning/views.py

@login_required
def knowledge_gap_finder(request):
    """
    Identifies and displays knowledge gaps.
    Can also be used as a targeted review session for a specific task.
    """
    review_task = None
    review_content = None

    task_id = request.GET.get('task_id')
    if task_id:
        try:
            # --- THIS IS THE CORRECTED LINE ---
            # We now check ownership by looking through the scheduled_day and study_plan
            review_task = ScheduledTask.objects.get(
                id=task_id, 
                scheduled_day__study_plan__user=request.user
            )
            # --- END OF CORRECTION ---

            if review_task.lesson:
                cheat_sheet = ai_utils.generate_cheat_sheet(review_task.lesson.title)
                review_content = {
                    'lesson': review_task.lesson,
                    'cheat_sheet': cheat_sheet
                }
        except ScheduledTask.DoesNotExist:
            messages.error(request, "The requested review task was not found.")
            return redirect('dashboard')

    if request.method == 'POST':
        new_gaps_data = ai_utils.find_knowledge_gaps(request.user)
        if new_gaps_data and new_gaps_data.get('weak_topics'):
            KnowledgeGap.objects.create(
                user=request.user,
                gap_details={'weak_topics': new_gaps_data['weak_topics']},
                ai_summary=new_gaps_data['ai_summary']
            )
        else:
            messages.info(request, "Analysis complete. No significant knowledge gaps found!")
        return redirect('knowledge_gap_finder')

    gaps = KnowledgeGap.objects.filter(user=request.user).order_by('-identified_date')
    
    context = {
        'gaps': gaps,
        'review_content': review_content,
        'review_task': review_task
    }
    return render(request, 'learning/knowledge_gap_finder.html', context)

# In learning/views.py
# In learning/views.py
# learning/views.py
# ... (all your other imports should be at the top) ...
# learning/views.py
# ... (all other necessary imports) ...
# In learning/views.py, replace this entire function

@login_required
def mock_exam_center_view(request):
    study_plans = StudyPlan.objects.filter(user=request.user)

    if request.method == 'POST':
        plan_id = request.POST.get('study_plan')
        question_type = request.POST.get('question_type', 'mixed')
        
        if not plan_id:
            messages.error(request, "Please select a study plan.")
            return redirect('mock_exam_center')
            
        study_plan = get_object_or_404(StudyPlan, id=plan_id, user=request.user)
        all_lessons = Lesson.objects.filter(module__study_plan=study_plan)
        if not all_lessons.exists():
            messages.error(request, "This study plan has no lessons to generate an exam from.")
            return redirect('mock_exam_center')

        user_profile = []
        for lesson in all_lessons:
            mastery_score = 0.0
            try:
                # --- THIS IS THE CORRECTED LINE ---
                # We now filter by the 'lesson' object itself, not the old 'concept_key'.
                mastery = Mastery.objects.get(user=request.user, lesson=lesson)
                # --- END OF CORRECTION ---
                mastery_score = mastery.mastery_score
            except Mastery.DoesNotExist:
                pass
            
            incorrect_answers = UserAnswer.objects.filter(
                quiz_attempt__user=request.user, question__lesson=lesson, is_correct=False
            ).values_list('question__category', flat=True)
            
            user_profile.append({
                'lesson_title': lesson.title,
                'mastery_score': f"{mastery_score:.0%}",
                'weak_categories': dict(Counter(incorrect_answers))
            })
        
        # The rest of your view logic for generating the exam is correct and can remain the same.
        from .ai_utils import generate_adaptive_mock_exam
        
        try:
            exam_json_str = generate_adaptive_mock_exam(user_profile, question_type=question_type)
            cleaned_json = exam_json_str.replace('```json', '').replace('```', '').strip()
            exam_data = json.loads(cleaned_json)
            
            lesson_map = {lesson.title: lesson for lesson in all_lessons}
            exam_title = f"Mock Exam for {study_plan.title}"
            
            # Updated to use the new quiz_type field
            new_exam = Quiz.objects.create(
                user=request.user, title=exam_title, quiz_type='mock_exam'
            )
            
            weak_topics_summary = [p['lesson_title'] for p in user_profile if float(p['mastery_score'].strip('%')) < 60]
            if weak_topics_summary:
                debug_message = f"This mock exam is personalized for you. It will focus more on your weaker topics, such as: {', '.join(weak_topics_summary[:3])}."
            else:
                debug_message = "This mock exam is personalized for you. You've shown strong mastery, so expect a challenging set of questions!"
            
            request.session['ai_insight_message'] = debug_message

            for q_data in exam_data['questions']:
                question_topic_title = q_data.get('lesson') 
                question_text = q_data.get('question')
                question_type_ai = q_data.get('question_type', 'multiple-choice')
                question_lesson = lesson_map.get(question_topic_title)
                category = q_data.get('category', 'CONCEPTUAL').upper()
                grading_rubric = q_data.get('answer') if question_type_ai == 'open-ended' else None

                question = Question.objects.create(
                    quiz=new_exam, lesson=question_lesson, text=question_text, 
                    category=category, question_type=question_type_ai, grading_rubric=grading_rubric
                )
                
                if question_type_ai == 'multiple-choice':
                    correct_answer_text = q_data.get('answer')
                    for ans_text in q_data.get('options', []):
                        is_correct = (ans_text == correct_answer_text)
                        Answer.objects.create(question=question, text=ans_text, is_correct=is_correct)
            
            messages.success(request, f"Your mock exam '{exam_title}' has been generated!")
            return redirect('take_quiz', quiz_id=new_exam.id)
            
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            messages.error(request, f"The AI failed to generate a valid exam. The error was: {e}. Please try again.")
            return redirect('mock_exam_center')

    return render(request, 'learning/mock_exam_center.html', {'study_plans': study_plans})


# Add this new view in learning/views.py under the QUIZ & ASSESSMENT section
# In learning/views.py, replace the entire automated_quiz_view function

@login_required
def automated_quiz_view(request, quiz_type):
    """
    Generates and displays automated quizzes with intelligent and distinct fallbacks.
    """
    user = request.user
    all_user_lessons = Lesson.objects.filter(module__study_plan__user=user)
    lessons_for_quiz = []
    
    if not all_user_lessons.exists():
        messages.info(request, "You need to have lessons in your study plan to generate a quiz.")
        return redirect('dashboard')

    if quiz_type == 'daily':
        # Daily quiz logic remains the same: find yesterday's completed lessons.
        yesterday = timezone.localdate() - timedelta(days=1)
        progress_records = UserLessonProgress.objects.filter(
            user=user, completed_at__date=yesterday
        ).select_related('lesson')
        
        if progress_records:
            lessons_for_quiz = [progress.lesson for progress in progress_records][:5]

    elif quiz_type == 'surprise':
        # Primary goal: find the 3 lessons with the lowest mastery scores.
        low_mastery_records = Mastery.objects.filter(user=user, lesson__isnull=False).order_by('mastery_score')[:3]
        
        if low_mastery_records.exists():
            lessons_for_quiz = [mastery.lesson for mastery in low_mastery_records]
        else:
            # --- ROBUST FALLBACK for Surprise Quiz ---
            # If no mastery scores, pick 3 truly random lessons.
            lesson_ids = list(all_user_lessons.values_list('id', flat=True))
            if len(lesson_ids) >= 3:
                random_lesson_ids = random.sample(lesson_ids, 3)
                lessons_for_quiz = list(Lesson.objects.filter(id__in=random_lesson_ids))
            else: # If user has less than 3 lessons total
                lessons_for_quiz = list(all_user_lessons)

            
    elif quiz_type == 'weekly':
        # Primary goal: find 5 lessons that haven't been tested recently.
        # This works even if mastery records exist but last_tested_date is null.
        lessons_with_mastery = Mastery.objects.filter(user=user, lesson__isnull=False)

        if lessons_with_mastery.exists():
             # Order by last tested date, nulls first (least recently tested)
            lessons_for_quiz = list(all_user_lessons.order_by('mastery__last_tested_date')[:5])
        else:
            # --- DISTINCT FALLBACK for Weekly Quiz ---
            # If no mastery records exist at all, just pick the first 5 lessons in the plan.
            lessons_for_quiz = list(all_user_lessons.order_by('module__order', 'order')[:5])


    # --- Centralized Check & Generation ---
    if not lessons_for_quiz:
        messages.info(request, f"We couldn't find enough topics for a '{quiz_type}' quiz right now. Have you completed any lessons recently?")
        return redirect('dashboard')

    quiz = generate_automated_quiz(user, quiz_type, lessons_for_quiz)

    if not quiz:
        messages.error(request, "Sorry, the AI had trouble creating your quiz. Please try again in a moment.")
        return redirect('dashboard')

    request.session['ai_insight_message'] = quiz.ai_insight
    return redirect('take_quiz', quiz_id=quiz.id)

# In learning/views.py, add this new view

@login_required
def run_maintenance_view(request):
    """
    A view to manually trigger the daily maintenance for the logged-in user.
    """
    # Call the AI utility function for the current user
    perform_daily_schedule_maintenance(request.user)
    
    # Add a success message to be displayed on the page
    messages.success(request, "AI schedule maintenance has been run successfully! Check for new notifications.")
    
    # Redirect back to the dashboard
    return redirect('dashboard')

# In learning/views.py, add this new section and view

# =============================================================================
# PRE-EXAM FEATURES
# =============================================================================
# In learning/views.py, replace the entire PRE-EXAM FEATURES section

# =============================================================================
# PRE-EXAM FEATURES
# =============================================================================

@login_required
def power_pack_view(request):
    """
    Lists existing Power Packs and provides a form to generate a new one.
    """
    # Get all study plans for the user to choose from
    study_plans = StudyPlan.objects.filter(user=request.user)
    # Get all previously generated power packs for this user
    power_packs = PowerPack.objects.filter(user=request.user).order_by('-created_at')

    if request.method == 'POST':
        plan_id = request.POST.get('study_plan')
        if plan_id:
            plan = get_object_or_404(StudyPlan, id=plan_id, user=request.user)
            # Call the AI function which now saves the pack to the DB
            new_pack = generate_pre_exam_power_pack(request.user, plan)
            
            if new_pack:
                messages.success(request, "Your new Power Pack has been successfully generated!")
                return redirect('power_pack_detail', pack_id=new_pack.id)
            else:
                messages.error(request, "Could not generate the Power Pack. Make sure the study plan has lessons with content.")
        else:
            messages.error(request, "Please select a study plan.")
        
        return redirect('power_pack')

    context = {
        'study_plans': study_plans,
        'power_packs': power_packs,
    }
    return render(request, 'learning/power_pack_list.html', context)


@login_required
def power_pack_detail_view(request, pack_id):
    """
    Displays the content of a specific Power Pack.
    """
    power_pack = get_object_or_404(PowerPack, id=pack_id, user=request.user)
    
    # Convert markdown content to HTML for display
    html_content = markdown2.markdown(power_pack.content, extras=['fenced-code-blocks', 'tables'])
    
    context = {
        'power_pack': power_pack,
        'html_content': html_content
    }
    return render(request, 'learning/power_pack_detail.html', context)

# In learning/views.py, add this new section and remove the old flashcard views

# =============================================================================
# REVISION CARD SYSTEM
# =============================================================================
# In learning/views.py

@login_required
def revision_card_dashboard_view(request):
    """
    Main dashboard for the Revision Card center. Displays daily review decks,
    a full library of all cards, and allows users to generate new cards.
    """
    if request.method == 'POST':
        # This POST logic for creating cards remains the same
        lesson_id = request.POST.get('lesson_id')
        if lesson_id:
            lesson = get_object_or_404(Lesson, id=lesson_id, module__study_plan__user=request.user)
            card, created = RevisionCard.objects.get_or_create(user=request.user, lesson=lesson)
            if created:
                card_content = ai_utils.generate_revision_card_content(lesson)
                card.content = card_content
                card.save()
                messages.success(request, f"Successfully generated a new revision card for '{lesson.title}'.")
            else:
                messages.info(request, f"You already have a revision card for '{lesson.title}'.")
            return redirect('revision_card_dashboard')

    # Fetch the daily and upcoming review decks
    decks = ai_utils.get_daily_revision_deck(request.user)
    
    # Find topics with mastery < 80% to suggest for new card creation
    suggestion_topics = Mastery.objects.filter(
        user=request.user, 
        mastery_score__lt=0.8,
        lesson__isnull=False
    ).select_related('lesson').order_by('mastery_score')

    # --- NEW: Fetch ALL revision cards for the library view ---
    all_revision_cards = RevisionCard.objects.filter(user=request.user).select_related('lesson').order_by('lesson__title')

    context = {
        'due_cards': decks['due_cards'],
        'upcoming_cards': decks['upcoming_cards'],
        'suggestion_topics': suggestion_topics,
        'all_revision_cards': all_revision_cards, # <-- Pass the new data to the template
    }
    return render(request, 'learning/revision_card_dashboard.html', context)


@login_required
def revision_card_review_session_view(request):
    """
    The main interface for reviewing a deck of revision cards.
    """
    card_ids_str = request.GET.get('cards', '')
    if not card_ids_str:
        messages.error(request, "No cards were selected for this review session.")
        return redirect('revision_card_dashboard')

    card_ids = [int(id) for id in card_ids_str.split(',') if id.isdigit()]
    
    review_cards = RevisionCard.objects.filter(id__in=card_ids, user=request.user).select_related('lesson')
    
    # Convert markdown content to HTML for display in the template
    for card in review_cards:
        card.html_content = markdown2.markdown(card.content, extras=['fenced-code-blocks', 'tables'])

    context = {
        'review_cards': review_cards,
        'deck_title': request.GET.get('title', 'Review Session')
    }
    return render(request, 'learning/revision_card_review_session.html', context)


@login_required
def process_revision_card_feedback_view(request, card_id):
    """
    Processes user feedback on a revision card and updates its spaced repetition schedule via AJAX.
    """
    if request.method == 'POST':
        card = get_object_or_404(RevisionCard, id=card_id, user=request.user)
        feedback = request.POST.get('feedback') # 'good' or 'again'

        if feedback == 'good':
            # User knows it well, increase the interval
            new_interval = card.last_interval_days * 2.5
        else: # 'again'
            # User struggled, reset the interval to 1 day
            new_interval = 1

        card.last_interval_days = int(new_interval)
        card.next_review_date = timezone.now().date() + timedelta(days=max(1, int(new_interval)))
        card.save()

        return JsonResponse({'status': 'ok', 'next_review_in_days': card.last_interval_days})

    return JsonResponse({'status': 'error'}, status=400)