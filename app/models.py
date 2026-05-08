from django.db import models

class NewsRaw(models.Model):
    """新闻原始数据表"""
    title = models.CharField('原始标题', max_length=255, blank=True, null=True)
    content = models.TextField('原始内容', blank=True, null=True)
    source = models.CharField('来源', max_length=100, blank=True, null=True)
    url = models.CharField('链接', max_length=255, unique=True)
    category = models.CharField('分类', max_length=100, blank=True, null=True)
    publish_date = models.DateTimeField('发布时间', blank=True, null=True)
    read_count = models.IntegerField('阅读量', default=0, blank=True, null=True)
    comment_count = models.IntegerField('评论数', default=0, blank=True, null=True)
    like_count = models.IntegerField('点赞数', default=0, blank=True, null=True)
    create_time = models.DateTimeField('创建时间', auto_now_add=True)

    def __str__(self):
        return self.title or self.url

    class Meta:
        verbose_name = '新闻原始数据'
        verbose_name_plural = '新闻原始数据'
        db_table = 'app_newsraw'
        unique_together = ('url',)
        indexes = [
            models.Index(fields=['url']),
            models.Index(fields=['source']),
            models.Index(fields=['publish_date']),
            models.Index(fields=['category']),
        ]

class NewsSentimentAnalysis(models.Model):
    """新闻情感分析原始数据表"""
    title_clean = models.CharField('清洗后标题', max_length=255, blank=True, null=True)
    content_clean = models.TextField('清洗后内容', blank=True, null=True)
    title_standard = models.CharField('标准化标题', max_length=255, blank=True, null=True)
    content_standard = models.TextField('标准化内容', blank=True, null=True)
    source = models.CharField('来源', max_length=100, blank=True, null=True)
    category = models.CharField('分类', max_length=100, blank=True, null=True)
    publish_date = models.DateTimeField('发布时间', blank=True, null=True)
    url = models.CharField('链接', max_length=255, unique=True)
    read_count = models.IntegerField('阅读量', default=0, blank=True, null=True)
    comment_count = models.IntegerField('评论数', default=0, blank=True, null=True)
    like_count = models.IntegerField('点赞数', default=0, blank=True, null=True)
    title_len = models.IntegerField('标题长度', default=0, blank=True, null=True)
    content_len = models.IntegerField('内容长度', default=0, blank=True, null=True)
    content_token_len = models.IntegerField('内容Token数', default=0, blank=True, null=True)
    sentiment = models.CharField('情感', max_length=20, blank=True, null=True)
    sentiment_score = models.FloatField('情感得分', null=True, blank=True)
    create_time = models.DateTimeField('创建时间', auto_now_add=True)

    def __str__(self):
        return self.title_clean or self.url

    def get_sentiment_display(self):
        sentiment_choices = {
            'positive': '正面',
            'negative': '负面',
            'neutral': '中性'
        }
        return sentiment_choices.get(self.sentiment, '未知')

    class Meta:
        verbose_name = '新闻情感分析'
        verbose_name_plural = '新闻情感分析'
        db_table = 'app_newssentimentanalysis'
        unique_together = ('url',)
        indexes = [
            models.Index(fields=['url']),
            models.Index(fields=['source']),
            models.Index(fields=['publish_date']),
            models.Index(fields=['category']),
        ]


class OverallSentiment(models.Model):
    """整体情感统计"""
    total_news = models.IntegerField(default=0)
    positive_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)
    negative_count = models.IntegerField(default=0)
    positive_pct = models.FloatField(default=0.0)
    neutral_pct = models.FloatField(default=0.0)
    negative_pct = models.FloatField(default=0.0)
    avg_sentiment_score = models.FloatField(default=0.0)
    create_time = models.DateTimeField(auto_now=True)

    def __str__(self):
        return "整体情感统计"

    class Meta:
        db_table = 'overall_sentiment'


class SourceSentiment(models.Model):
    """来源情感统计"""
    source = models.CharField(max_length=100)
    total_count = models.IntegerField(default=0)
    positive_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)
    negative_count = models.IntegerField(default=0)
    avg_sentiment_score = models.FloatField(default=0.0)

    def __str__(self):
        return self.source

    class Meta:
        db_table = 'source_sentiment'


class CategorySentiment(models.Model):
    """分类情感统计"""
    category = models.CharField(max_length=100)
    total_count = models.IntegerField(default=0)
    positive_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)
    negative_count = models.IntegerField(default=0)
    avg_sentiment_score = models.FloatField(default=0.0)

    def __str__(self):
        return self.category

    class Meta:
        db_table = 'category_sentiment'


class DateSentiment(models.Model):
    """每日情感趋势"""
    date_str = models.CharField(max_length=20)
    total_count = models.IntegerField(default=0)
    positive_count = models.IntegerField(default=0)
    neutral_count = models.IntegerField(default=0)
    negative_count = models.IntegerField(default=0)
    avg_sentiment_score = models.FloatField(default=0.0)

    def __str__(self):
        return self.date_str

    class Meta:
        db_table = 'date_sentiment'


class ScoreDistribution(models.Model):
    """情感得分分布"""
    score_range = models.CharField(max_length=30)
    count = models.IntegerField(default=0)

    def __str__(self):
        return self.score_range

    class Meta:
        db_table = 'score_distribution'


class SentimentKeywords(models.Model):
    """情感关键词"""
    keyword = models.CharField(max_length=50)
    frequency = models.IntegerField(default=0)
    sentiment = models.CharField(max_length=20, default='all')

    def __str__(self):
        return self.keyword

    class Meta:
        db_table = 'sentiment_keywords'