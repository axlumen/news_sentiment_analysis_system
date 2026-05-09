import json
import os
import io
import sys
from contextlib import redirect_stdout
from datetime import datetime

from config import Config
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.cache import cache_page
from django.views.decorators.csrf import csrf_exempt

from data.predict_sentiment import SentimentPredictor
from .models import CategorySentiment, SourceSentiment, NewsSentimentAnalysis, OverallSentiment, DateSentiment, \
    ScoreDistribution, SentimentKeywords


# 整体情感统计 - 使用缓存
@csrf_exempt
@cache_page(Config.cache_timeout)
def api_overall(request):
    try:
        data = OverallSentiment.objects.order_by('-id').first()
        if data:
            result = {
                'id': data.id,
                'total_news': data.total_news,
                'positive_count': data.positive_count,
                'neutral_count': data.neutral_count,
                'negative_count': data.negative_count,
                'positive_pct': data.positive_pct,
                'neutral_pct': data.neutral_pct,
                'negative_pct': data.negative_pct,
                'avg_sentiment_score': data.avg_sentiment_score,
                'create_time': data.create_time.isoformat() if data.create_time else None
            }
            return JsonResponse(result, safe=False)
        return JsonResponse({}, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# 日期情感趋势 - 使用缓存
@csrf_exempt
@cache_page(Config.cache_timeout)
def api_date(request):
    try:
        data = DateSentiment.objects.order_by('date_str')
        result = [{
            'id': item.id,
            'date_str': item.date_str,
            'total_count': item.total_count,
            'positive_count': item.positive_count,
            'neutral_count': item.neutral_count,
            'negative_count': item.negative_count,
            'avg_sentiment_score': item.avg_sentiment_score
        } for item in data]
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse([], safe=False)


# 分类情感统计 - 使用缓存和ORM
@csrf_exempt
@cache_page(Config.cache_timeout)
def api_category(request):
    try:
        data = CategorySentiment.objects.all()[:10]
        result = [{
            'id': item.id,
            'category': item.category,
            'total_count': item.total_count,
            'positive_count': item.positive_count,
            'neutral_count': item.neutral_count,
            'negative_count': item.negative_count,
            'avg_sentiment_score': item.avg_sentiment_score
        } for item in data]
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse([], safe=False)


# 来源情感统计 - 使用缓存和ORM
@csrf_exempt
@cache_page(Config.cache_timeout)
def api_source(request):
    try:
        data = SourceSentiment.objects.order_by('-avg_sentiment_score')[:10]
        result = [{
            'id': item.id,
            'source': item.source,
            'total_count': item.total_count,
            'positive_count': item.positive_count,
            'neutral_count': item.neutral_count,
            'negative_count': item.negative_count,
            'avg_sentiment_score': item.avg_sentiment_score
        } for item in data]
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse([], safe=False)


# 情感得分分布 - 使用缓存和ORM
@csrf_exempt
@cache_page(Config.cache_timeout)
def api_score(request):
    try:
        data = ScoreDistribution.objects.all()
        result = [{
            'id': item.id,
            'score_range': item.score_range,
            'count': item.count
        } for item in data]
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse([], safe=False)


# 关键词（词云专用，返回 name + value）- 使用缓存
@csrf_exempt
@cache_page(Config.cache_timeout)
def api_keywords(request):
    try:
        data = SentimentKeywords.objects.order_by('-frequency')[:50]
        result = [{"name": item.keyword, "value": item.frequency} for item in data]
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse([], safe=False)


@login_required
def index(request):
    """
    首页视图
    """
    # 获取整体情感统计数据
    try:
        overall_data = OverallSentiment.objects.order_by('-id').first()
        if overall_data:
            sentiment_stats = {
                'total': overall_data.total_news,
                'positive': overall_data.positive_count,
                'neutral': overall_data.neutral_count,
                'negative': overall_data.negative_count,
                'positive_percent': round(overall_data.positive_pct, 1),
                'neutral_percent': round(overall_data.neutral_pct, 1),
                'negative_percent': round(overall_data.negative_pct, 1)
            }
        else:
            sentiment_stats = {
                'total': 0,
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'positive_percent': 0,
                'neutral_percent': 0,
                'negative_percent': 0
            }
    except Exception:
        sentiment_stats = {
            'total': 0,
            'positive': 0,
            'neutral': 0,
            'negative': 0,
            'positive_percent': 0,
            'neutral_percent': 0,
            'negative_percent': 0
        }

    # 获取近7天情感趋势数据
    try:
        trend_data = DateSentiment.objects.order_by('-date_str')[:7]
        trend_data = sorted(trend_data, key=lambda x: x.date_str)  # 按日期升序排序
    except Exception:
        trend_data = []

    # 获取分类情感数据
    try:
        category_data = CategorySentiment.objects.all()[:10]
    except Exception:
        category_data = []

    context = {
        'sentiment_stats': sentiment_stats,
        'trend_data': trend_data,
        'category_data': category_data
    }

    return render(request, 'index.html', context)


@login_required
def news_list(request):
    """
    新闻列表视图
    """
    # 获取筛选参数
    keyword = request.GET.get('keyword', '')
    sentiment = request.GET.get('sentiment', '')
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')
    category = request.GET.get('category', '')

    # 基础查询
    queryset = NewsSentimentAnalysis.objects.all().order_by('-create_time')

    # 关键词筛选
    if keyword:
        queryset = queryset.filter(
            title_clean__icontains=keyword
        )

    # 情感倾向筛选
    if sentiment:
        queryset = queryset.filter(sentiment=sentiment)

    # 板块筛选
    if category:
        queryset = queryset.filter(category=category)

    # 时间范围筛选
    if start_date:
        queryset = queryset.filter(publish_date__gte=start_date)
    if end_date:
        queryset = queryset.filter(publish_date__lte=end_date)

    # 分页处理
    paginator = Paginator(queryset, 10)  # 每页10条
    page = request.GET.get('page', 1)

    try:
        news_data = paginator.page(page)
        for news in news_data:
            news.sentiment_score_percent = round(news.sentiment_score * 100) if news.sentiment_score else 0
    except PageNotAnInteger:
        news_data = paginator.page(1)
        for news in news_data:
            news.sentiment_score_percent = round(news.sentiment_score * 100) if news.sentiment_score else 0
    except EmptyPage:
        news_data = paginator.page(paginator.num_pages)
        for news in news_data:
            news.sentiment_score_percent = round(news.sentiment_score * 100) if news.sentiment_score else 0

    context = {
        'news_list': news_data,
        'keyword': keyword,
        'sentiment': sentiment,
        'category': category,
        'start_date': start_date,
        'end_date': end_date,
        'paginator': paginator
    }
    return render(request, 'analysis/news_list.html', context)


@login_required
def sentiment_dashboard(request):
    """
    情感仪表盘：词云 + 情感倾向柱状图
    """
    return render(request, 'analysis/sentiment_dashboard.html')


@login_required
def sentiment_analysis(request):
    """
    情感分析页面视图
    """
    return render(request, 'analysis/sentiment_analysis.html')


@csrf_exempt
def analyze_text(request):
    """文本情感分析接口"""
    if request.method == 'POST':
        try:
            # 1. 解析请求数据
            data = json.loads(request.body)
            text = data.get('text', '')

            if not text:
                return JsonResponse({'error': '请输入要分析的文本'}, status=400)

            # 2. 构建模型和词表路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'model', 'checkpoints', 'best_model.pth')
            vocab_path = os.path.join(base_dir, 'model', 'dataset', 'processed', 'vocab.pkl')

            # 3. 检查文件是否存在
            if not os.path.exists(model_path):
                return JsonResponse({'error': f'模型文件不存在: {model_path}'}, status=500)
            if not os.path.exists(vocab_path):
                return JsonResponse({'error': f'词表文件不存在: {vocab_path}'}, status=500)

            # 4. 初始化预测器
            predictor = SentimentPredictor(model_path, vocab_path)
            
            # 5. 执行预测
            result = predictor.predict_single(text)

            # 6. 返回结果
            return JsonResponse({
                'sentiment': result['sentiment'],
                'score': result['sentiment_score']
            })
        except json.JSONDecodeError as e:
            return JsonResponse({'error': f'JSON解析错误: {str(e)}'}, status=400)
        except Exception as e:
            # 记录详细错误信息
            import traceback
            error_msg = f'错误类型: {type(e).__name__}, 错误信息: {str(e)}, 堆栈: {traceback.format_exc()}'
            print(f"情感分析错误: {error_msg}")
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': '只支持POST请求'}, status=405)


@csrf_exempt
def api_overall_sentiment(request):
    """整体情感统计接口"""
    try:
        data = OverallSentiment.objects.order_by('-id').first()
        if not data:
            return JsonResponse({"total_news": 0, "positive_count": 0, "neutral_count": 0, "negative_count": 0,
                                "positive_pct": 0, "neutral_pct": 0, "negative_pct": 0, "avg_sentiment_score": 0})
        return JsonResponse({
            "total_news": data.total_news,
            "positive_count": data.positive_count,
            "neutral_count": data.neutral_count,
            "negative_count": data.negative_count,
            "positive_pct": data.positive_pct,
            "neutral_pct": data.neutral_pct,
            "negative_pct": data.negative_pct,
            "avg_sentiment_score": data.avg_sentiment_score
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_daily_sentiment(request):
    """每日情感趋势接口"""
    try:
        data = DateSentiment.objects.all().order_by('date_str')
        result = []
        for item in data:
            result.append({
                "date": item.date_str,
                "positive_count": item.positive_count,
                "neutral_count": item.neutral_count,
                "negative_count": item.negative_count,
                "total_count": item.total_count,
                "avg_sentiment_score": item.avg_sentiment_score
            })
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_category_sentiment(request):
    """分类情感接口"""
    try:
        data = CategorySentiment.objects.all()[:10]
        result = []
        for item in data:
            result.append({
                "category": item.category,
                "positive_count": item.positive_count,
                "neutral_count": item.neutral_count,
                "negative_count": item.negative_count,
                "total_count": item.total_count,
                "avg_sentiment_score": item.avg_sentiment_score
            })
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_source_sentiment(request):
    """来源情感接口"""
    try:
        data = SourceSentiment.objects.all()[:10]
        result = []
        for item in data:
            result.append({
                "source": item.source,
                "avg_sentiment_score": item.avg_sentiment_score,
                "total_count": item.total_count,
                "positive_count": item.positive_count,
                "neutral_count": item.neutral_count,
                "negative_count": item.negative_count
            })
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_score_distribution(request):
    """情感得分分布接口"""
    try:
        data = ScoreDistribution.objects.all()
        result = []
        for item in data:
            result.append({
                "score_range": item.score_range,
                "count": item.count
            })
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def api_sentiment_keywords(request):
    """情感关键词接口（兼容旧模板）"""
    try:
        data = SentimentKeywords.objects.order_by('-frequency')[:20]
        result = []
        for item in data:
            result.append({
                'name': item.keyword,  # ECharts 词云需要 'name' 字段
                'value': item.frequency,  # ECharts 词云需要 'value' 字段
                'word': item.keyword,  # 保持兼容旧模板
                'count': item.frequency,  # 保持兼容旧模板
                'keyword': item.keyword,  # 兼容dashboard.html
                'frequency': item.frequency,  # 兼容dashboard.html
                'sentiment': item.sentiment or 'all',  # 添加情感类型字段
            })
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse([], safe=False)


@csrf_exempt
def api_news_list(request):
    """新闻列表接口"""
    try:
        news_data = NewsSentimentAnalysis.objects.all().order_by('-create_time')[:20]
        result = []
        for news in news_data:
            sentiment_score_norm = news.sentiment_score if news.sentiment_score is not None else 0
            result.append({
                'id': news.id,
                'title': news.title_clean,
                'source': news.source,
                'publish_date': news.publish_date.strftime('%Y-%m-%d') if news.publish_date else None,
                'sentiment': news.sentiment,
                'sentiment_score': news.sentiment_score,
                'sentiment_score_norm': sentiment_score_norm
            })
        return JsonResponse(result, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def hot_search_wordcloud(request):
    """
    热搜词云页面（对接你的模板）
    """
    return render(request, 'analysis/hot_search_wordcloud.html')


@login_required
def statistics(request):
    return redirect('app:news_list')


@login_required
def dashboard(request):
    return render(request, 'analysis/dashboard.html')


@login_required
def data_update(request):
    """
    数据更新页面视图
    """
    return render(request, 'analysis/data_update.html')


@csrf_exempt
@login_required
def api_run_pipeline(request):
    """
    运行数据全流程管道API
    执行：爬虫 -> 预处理 -> 预测 -> 分析
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': '只支持POST请求'}, status=405)
    
    try:
        import io
        import sys
        from datetime import datetime
        
        # 捕获输出
        output_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = output_buffer
        
        # 直接调用data.main的main函数
        print("=" * 80)
        print("新闻情感分析系统 - 自动化数据流管道")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 导入并调用main函数
        from data.main import main
        main()
        
        # 恢复输出
        sys.stdout = old_stdout
        output_log = output_buffer.getvalue()
        output_buffer.close()
        
        return JsonResponse({
            'success': True,
            'message': '数据全流程执行完成！',
            'log': output_log,
            'completed': 4,
            'total': 4
        })
        
    except Exception as e:
        import traceback
        error_msg = f"执行出错: {str(e)}\n{traceback.format_exc()}"
        return JsonResponse({
            'success': False,
            'message': f'执行失败: {str(e)}',
            'log': error_msg
        }, status=500)


@csrf_exempt
@login_required
def api_run_selected_steps(request):
    """
    运行选中的步骤API
    支持选择执行：爬虫、数据清洗、情感分析、统计更新
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': '只支持POST请求'}, status=405)
    
    try:
        # 解析请求数据
        data = json.loads(request.body)
        steps = data.get('steps', [])
        
        if not steps:
            return JsonResponse({'success': False, 'message': '请至少选择一个步骤'}, status=400)

        # 验证步骤名称合法性
        valid_steps = {'spider', 'clean', 'analysis', 'statistics'}
        invalid = set(steps) - valid_steps
        if invalid:
            return JsonResponse({'success': False, 'message': f'无效的步骤: {", ".join(invalid)}'}, status=400)

        # 使用 contextlib.redirect_stdout 安全捕获输出
        output_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer):
            print("=" * 80)
            print("新闻情感分析系统 - 选择步骤执行")
            print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"选中步骤: {', '.join(steps)}")
            print("=" * 80)
            
            # 导入步骤函数
            from data.main import run_spider, run_preprocess, run_predict, run_analysis
            
            # 步骤映射
            step_map = {
                'spider': ('新闻爬虫', run_spider),
                'clean': ('数据清洗', run_preprocess),
                'analysis': ('情感分析', run_predict),
                'statistics': ('统计更新', run_analysis)
            }
            
            # 按顺序执行选中的步骤
            step_order = ['spider', 'clean', 'analysis', 'statistics']
            results = []
            
            for step_key in step_order:
                if step_key in steps:
                    name, func = step_map[step_key]
                    print(f"\n{'=' * 60}")
                    print(f"执行步骤: {name}")
                    print('=' * 60)
                    success = func()
                    results.append((name, success))
                    if not success:
                        print(f"警告: {name} 执行失败")
            
            # 打印执行摘要
            print("\n" + "=" * 60)
            print("执行摘要")
            print("=" * 60)
            for name, success in results:
                status = "[OK] 成功" if success else "[FAIL] 失败"
                print(f"{name}: {status}")
            print("=" * 60)
            print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        output_log = output_buffer.getvalue()
        output_buffer.close()
        
        return JsonResponse({
            'success': True,
            'message': '步骤执行完成！',
            'log': output_log,
            'steps': steps,
            'results': results
        })
        
    except json.JSONDecodeError as e:
        return JsonResponse({'success': False, 'message': f'JSON解析错误: {str(e)}'}, status=400)
    except Exception as e:
        import traceback
        error_msg = f"执行出错: {str(e)}\n{traceback.format_exc()}"
        return JsonResponse({
            'success': False,
            'message': f'执行失败: {str(e)}',
            'log': error_msg
        }, status=500)