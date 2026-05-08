from django.urls import path
from app import views

app_name = 'app'
urlpatterns = [
    path('', views.index, name='index'),
    path('news/list/', views.news_list, name='news_list'),
    path('sentiment/analysis/', views.sentiment_analysis, name='sentiment_analysis'),
    path('hot_search_wordcloud/', views.hot_search_wordcloud, name='hot_search_wordcloud'),  # 词云页面
    path('dashboard/', views.dashboard, name='dashboard'),

    # ===================== 全量接口（与视图、模板完全匹配） =====================
    path('api/overall_sentiment/', views.api_overall_sentiment, name='api_overall_sentiment'),
    path('api/daily_sentiment/', views.api_daily_sentiment, name='api_daily_sentiment'),
    path('api/category_sentiment/', views.api_category_sentiment, name='api_category_sentiment'),
    path('api/source_sentiment/', views.api_source_sentiment, name='api_source_sentiment'),
    path('api/score_distribution/', views.api_score_distribution, name='api_score_distribution'),
    path('api/sentiment_keywords/', views.api_sentiment_keywords, name='api_sentiment_keywords'),  # 词云数据
    path('api/news_list/', views.api_news_list, name='api_news_list'),
    path('api/analyze_text/', views.analyze_text, name='analyze_text'),
    path('api/run_pipeline/', views.api_run_pipeline, name='api_run_pipeline'),  # 数据全流程运行
]