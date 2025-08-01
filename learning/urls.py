# In learning/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # =============================================================================
    # MAIN DASHBOARD & HOME ROUTES
    # =============================================================================
    path('', views.dashboard_view, name='dashboard'),
    path('home/', views.home_view, name='home'),
    path('profile/', views.profile_view, name='profile'),
    path('my-progress/', views.my_progress_view, name='my_progress'),

    # =============================================================================
    # STUDY PLAN MANAGEMENT ROUTES
    # =============================================================================
    path('generate-plan/', views.generate_plan_view, name='generate_plan'),
    path('save-plan/', views.save_plan_view, name='save_plan'),
    path('my-plans/', views.my_plans_view, name='my_plans'),
    path('plan/<int:plan_id>/', views.plan_detail_view, name='plan_detail'),

    # =============================================================================
    # LEARNING CONTENT & LESSON ROUTES
    # =============================================================================
    path('modules/', views.module_list_view, name='module_list'),
    path('lesson/<int:lesson_id>/', views.lesson_detail_view, name='lesson_detail'),
    path('lesson/<int:lesson_id>/complete/', views.mark_lesson_complete_view, name='mark_lesson_complete'),
    path('lesson/<int:lesson_id>/ask_ai/', views.ask_ai_view, name='ask_ai'),

    # =============================================================================
    # QUIZ & ASSESSMENT ROUTES
    # =============================================================================
    # --- NEW: Routes for our automated quizzes ---
    path('quiz/generate/<str:quiz_type>/', views.automated_quiz_view, name='automated_quiz'),
    
    # --- Consolidated and cleaned up quiz routes ---
    path('practice-center/', views.practice_center_view, name='practice_center'),
    path('quiz/<int:quiz_id>/', views.take_quiz_view, name='take_quiz'),
    path('quiz/<int:quiz_id>/submit/', views.submit_quiz_view, name='submit_quiz'),
    path('quiz/results/<int:attempt_id>/', views.quiz_results_view, name='quiz_results'),
    path('quiz-history/', views.quiz_history_view, name='quiz_history'),
    path('quiz-review/<int:attempt_id>/', views.quiz_review_view, name='quiz_review'),
    path('mock-exam-center/', views.mock_exam_center_view, name='mock_exam_center'),

    # =============================================================================
    # SPACED REPETITION & REVIEW ROUTES
    # =============================================================================
     path('flashcards/', views.flashcard_dashboard_view, name='flashcard_dashboard'),
    path('flashcards/set/<int:set_id>/', views.flashcard_set_view, name='flashcard_set_view'),
    path('flashcards/review/', views.flashcard_review_session_view, name='flashcard_review_session'),
    path('review-session/<int:task_id>/', views.review_session_view, name='review_session'),
    path('review/card/<int:user_flashcard_id>/', views.process_review_answer_view, name='process_review_answer'),
    path('knowledge-gaps/', views.knowledge_gap_finder, name='knowledge_gap_finder'),

     path('revision-cards/', views.revision_card_dashboard_view, name='revision_card_dashboard'),
    path('revision-cards/review/', views.revision_card_review_session_view, name='revision_card_review_session'),
    path('revision-cards/<int:card_id>/feedback/', views.process_revision_card_feedback_view, name='process_revision_card_feedback'),
    # --- END NEW ROUTES ---

    # =============================================================================
    # SCHEDULING & PLANNING ROUTES
    # =============================================================================
    path('scheduler/', views.scheduler_view, name='scheduler'),

    # =============================================================================
    # COMMUNITY & ADDITIONAL FEATURE ROUTES
    # =============================================================================
    path('projects/', views.project_generation_view, name='project_generation'),
    path('studypals/', views.studypals_view, name='studypals'),

    # =============================================================================
    # API ENDPOINTS (AJAX)
    # =============================================================================
    path('api/get-lesson-content/<int:lesson_id>/<str:content_type>/', views.get_lesson_content_view, name='get_lesson_content'),
    path('api/log-interaction/', views.log_interaction_view, name='log_interaction'),
    path('api/ask-tutor/', views.ask_tutor_view, name='ask_tutor'),
    path('run-maintenance/', views.run_maintenance_view, name='run_maintenance'),
    path('power-packs/', views.power_pack_view, name='power_pack_list'),
path('power-pack/<int:pack_id>/', views.power_pack_detail_view, name='power_pack_detail'),


]