"""
新闻情感分析系统 - 自动化数据流管道

整合爬虫、数据预处理、情感分析预测、统计分析的完整流程
实现一键运行全链路数据处理

使用方式:
    python data/main.py [options]
    
选项:
    --skip-spider       跳过爬虫步骤
    --skip-preprocess   跳过数据预处理步骤
    --skip-predict      跳过情感预测步骤
    --skip-analysis     跳过统计分析步骤
    --full              执行完整流程（包括清空旧数据）
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'news_sentiment_analysis.settings')
import django
django.setup()

from app.models import NewsSentimentAnalysis, NewsRaw


def print_step(step_num, total_steps, title):
    """打印步骤分隔线"""
    print("\n" + "=" * 80)
    print(f"步骤 {step_num}/{total_steps}: {title}")
    print("=" * 80)


def check_raw_data_count():
    """检查原始数据表的数据量"""
    try:
        raw_count = NewsRaw.objects.count()
        print(f"原始数据表 (app_newsraw) 数据量: {raw_count}")
        return raw_count
    except Exception as e:
        print(f"检查原始数据表失败: {e}")
        return 0


def check_data_status():
    """检查当前数据状态"""
    print("\n当前数据状态:")
    print("-" * 40)
    
    # 检查原始数据
    raw_count = check_raw_data_count()
    
    # 检查新闻总数
    total_news = NewsSentimentAnalysis.objects.count()
    print(f"新闻总数: {total_news}")
    
    # 检查已分析的新闻数
    analyzed_news = NewsSentimentAnalysis.objects.filter(
        sentiment__isnull=False
    ).count()
    print(f"已分析新闻数: {analyzed_news}")
    
    # 检查情感分布
    if analyzed_news > 0:
        positive = NewsSentimentAnalysis.objects.filter(sentiment='positive').count()
        neutral = NewsSentimentAnalysis.objects.filter(sentiment='neutral').count()
        negative = NewsSentimentAnalysis.objects.filter(sentiment='negative').count()
        print(f"  - 正面: {positive}")
        print(f"  - 中性: {neutral}")
        print(f"  - 负面: {negative}")
    
    print("-" * 40)
    return raw_count, total_news, analyzed_news


def run_spider():
    """运行新闻爬虫"""
    print("开始爬取新闻数据...")
    try:
        from data.news_spider import main as spider_main
        spider_main()
        return True
    except Exception as e:
        print(f"爬虫执行失败: {e}")
        return False


def run_preprocess():
    """运行数据预处理"""
    print("开始数据预处理...")
    try:
        # 检查是否有原始数据
        raw_count = check_raw_data_count()
        if raw_count == 0:
            print("⚠️ 原始数据表中没有数据，跳过预处理")
            return True
        
        from data.data_preprocess import main as preprocess_main
        preprocess_main()
        return True
    except Exception as e:
        print(f"数据预处理失败: {e}")
        return False


def run_predict():
    """运行情感预测"""
    print("开始情感分析预测...")
    try:
        # 检查是否有清洗后的数据
        total_news = NewsSentimentAnalysis.objects.count()
        if total_news == 0:
            print("⚠️ 没有待预测的数据，跳过情感预测")
            return True
        
        from data.predict_sentiment import main as predict_main
        predict_main()
        return True
    except Exception as e:
        print(f"情感预测失败: {e}")
        return False


def run_analysis():
    """运行统计分析"""
    print("开始生成统计报表...")
    try:
        # 检查是否有情感分析后的数据
        analyzed_news = NewsSentimentAnalysis.objects.filter(
            sentiment__isnull=False
        ).count()
        if analyzed_news == 0:
            print("⚠️ 没有情感分析数据，跳过统计分析")
            return True
        
        from data.news_analysis import create_stat_tables, analyze_and_save
        create_stat_tables()
        analyze_and_save()
        return True
    except Exception as e:
        print(f"统计分析失败: {e}")
        return False


def main(skip_spider=False, skip_preprocess=False, skip_predict=False, skip_analysis=False, full=False):
    """
    主函数
    参数说明：
    - skip_spider: 跳过爬虫步骤
    - skip_preprocess: 跳过数据预处理步骤
    - skip_predict: 跳过情感预测步骤
    - skip_analysis: 跳过统计分析步骤
    - full: 执行完整流程
    """
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("新闻情感分析系统 - 自动化数据流管道")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 检查初始数据状态
    raw_count, total_news, analyzed_news = check_data_status()
    
    # 确定执行步骤
    steps = []
    step_num = 0
    
    if not skip_spider:
        step_num += 1
        steps.append((step_num, "新闻爬虫", run_spider))
    
    if not skip_preprocess:
        step_num += 1
        steps.append((step_num, "数据预处理", run_preprocess))
    
    if not skip_predict:
        step_num += 1
        steps.append((step_num, "情感分析预测", run_predict))
    
    if not skip_analysis:
        step_num += 1
        steps.append((step_num, "统计分析", run_analysis))
    
    total_steps = len(steps)
    
    if total_steps == 0:
        print("\n没有需要执行的步骤")
        return
    
    # 执行各个步骤
    results = []
    for num, name, func in steps:
        print_step(num, total_steps, name)
        success = func()
        results.append((name, success))
        if not success:
            print(f"警告: {name} 执行失败，继续后续步骤...")
        
        # 每个步骤完成后检查数据状态
        print(f"\n步骤 {num} 完成后数据状态:")
        check_data_status()
    
    # 检查最终数据状态
    print_step(total_steps + 1, total_steps + 1, "最终数据状态检查")
    check_data_status()
    
    # 打印执行摘要
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("执行摘要")
    print("=" * 80)
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{name}: {status}")
    print("-" * 80)
    print(f"总耗时: {duration:.2f} 秒")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def main_with_args():
    """
    命令行入口函数，用于直接运行脚本
    """
    parser = argparse.ArgumentParser(description='新闻情感分析系统数据流管道')
    parser.add_argument('--skip-spider', action='store_true', help='跳过爬虫步骤')
    parser.add_argument('--skip-preprocess', action='store_true', help='跳过数据预处理步骤')
    parser.add_argument('--skip-predict', action='store_true', help='跳过情感预测步骤')
    parser.add_argument('--skip-analysis', action='store_true', help='跳过统计分析步骤')
    parser.add_argument('--full', action='store_true', help='执行完整流程')
    
    args = parser.parse_args()
    
    main(
        skip_spider=args.skip_spider,
        skip_preprocess=args.skip_preprocess,
        skip_predict=args.skip_predict,
        skip_analysis=args.skip_analysis,
        full=args.full
    )


if __name__ == "__main__":
    main_with_args()
