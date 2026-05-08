from django.contrib import admin
from .models import (
    NewsSentimentAnalysis,
    CategorySentiment,
    SourceSentiment,
    OverallSentiment,
    DateSentiment,
    ScoreDistribution,
    SentimentKeywords
)


# ========== 新闻原始数据 ==========
@admin.register(NewsSentimentAnalysis)
class NewsSentimentAnalysisAdmin(admin.ModelAdmin):
    list_display = ('title_clean', 'source', 'category', 'sentiment', 'sentiment_score', 'publish_date', 'create_time')
    search_fields = ('title_clean', 'content_clean', 'source', 'category')
    list_filter = ('sentiment', 'source', 'category', 'publish_date', 'create_time')
    readonly_fields = ('create_time',)
    list_per_page = 20
    ordering = ('-create_time',)


# ========== 分类情感 ==========
@admin.register(CategorySentiment)
class CategorySentimentAdmin(admin.ModelAdmin):
    list_display = ('category', 'total_count', 'positive_count', 'negative_count', 'neutral_count', 'avg_sentiment_score')
    search_fields = ('category',)
    list_per_page = 20
    ordering = ('-total_count',)


# ========== 来源情感 ==========
@admin.register(SourceSentiment)
class SourceSentimentAdmin(admin.ModelAdmin):
    list_display = ('source', 'total_count', 'positive_count', 'negative_count', 'neutral_count', 'avg_sentiment_score')
    list_filter = ('source',)
    search_fields = ('source',)
    list_per_page = 20
    ordering = ('-total_count',)


# ========== 整体情感 ==========
@admin.register(OverallSentiment)
class OverallSentimentAdmin(admin.ModelAdmin):
    list_display = ('total_news', 'positive_count', 'neutral_count', 'negative_count', 'avg_sentiment_score', 'create_time')
    list_per_page = 20


# ========== 每日情感趋势 ==========
@admin.register(DateSentiment)
class DateSentimentAdmin(admin.ModelAdmin):
    list_display = ('date_str', 'total_count', 'positive_count', 'neutral_count', 'negative_count', 'avg_sentiment_score')
    list_per_page = 20


# ========== 情感得分分布 ==========
@admin.register(ScoreDistribution)
class ScoreDistributionAdmin(admin.ModelAdmin):
    list_display = ('score_range', 'count')
    list_per_page = 20


# ========== 关键词 ==========
@admin.register(SentimentKeywords)
class SentimentKeywordsAdmin(admin.ModelAdmin):
    list_display = ('keyword', 'frequency', 'sentiment')
    search_fields = ('keyword',)
    list_per_page = 20