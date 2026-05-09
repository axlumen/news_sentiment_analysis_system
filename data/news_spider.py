import logging
import re
import time
import random
from datetime import datetime
import requests
import pymysql
from parsel import Selector
from fake_useragent import UserAgent
from dotenv import load_dotenv
from config import Config
from utils.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# -------------------------- 配置项 --------------------------
# MySQL配置
MYSQL_CONFIG = {
    "host": Config.host,
    "port": int(Config.port),
    "user": Config.user,
    "password": Config.password,
    "database": Config.database,
    "charset": Config.charset
}

# 【关键修复】绕过新浪安全策略 请求头
ua = UserAgent()
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Mode": "navigate"
}

# 爬虫控制
DELAY = (2, 4)
MAX_RETRY = 3
MAX_TOTAL_NEWS = 20  # 总爬取上限
MAX_CATEGORY_NEWS = 10  # 单分类上限

# -------------------------- MySQL --------------------------
class NewsMySQL:
    def __init__(self):
        self.config = MYSQL_CONFIG
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        try:
            self.conn = pymysql.connect(**MYSQL_CONFIG)
            self.cursor = self.conn.cursor(pymysql.cursors.DictCursor)
            print("✅ MySQL连接成功")
        except Exception as e:
            print(f"❌ MySQL连接失败：{e}")
            raise

    def close(self):
        if self.cursor: self.cursor.close()
        if self.conn: self.conn.close()

    def check_duplicate(self, url):
        sql = "SELECT 1 FROM app_newsraw WHERE url = %s LIMIT 1"
        self.cursor.execute(sql, (url,))
        return self.cursor.fetchone() is not None

    def insert_news(self, news_data):
        """直接插入Django模型表，字段名与模型一致"""
        if self.check_duplicate(news_data["url"]):
            print(f"⏭️ 已存在：{news_data['title']}")
            return False

        sql = """
        INSERT INTO app_newsraw 
        (title, content, source, url, category, publish_date, read_count, comment_count, like_count, create_time)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
        title = VALUES(title),
        content = VALUES(content),
        source = VALUES(source),
        category = VALUES(category),
        publish_date = VALUES(publish_date)
        """
        try:
            self.cursor.execute(sql, (
                news_data["title"], news_data["content"], news_data["source"], news_data["url"],
                news_data["category"], news_data["publish_date"], news_data["read_count"],
                news_data["comment_count"], news_data["like_count"],
                datetime.now()
            ))
            self.conn.commit()
            print(f"✅ 入库：{news_data['title']}")
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"❌ 入库失败：{e}")
            return False

# TextCleaner imported from utils.text_cleaner
class _TextCleanerLegacy:
    @staticmethod
    def clean_text(text):
        if not text: return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】《》、·\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def parse_date(s):
        if not s: return None
        try:
            s = re.sub(r'[年月日/]', '-', s)
            match = re.search(r'\d{4}-\d{1,2}-\d{1,2}', s)
            if match:
                return datetime.strptime(match.group(), "%Y-%m-%d").date()
        except:
            return None

# -------------------------- 新浪爬虫 --------------------------
class SinaSpider:
    def __init__(self, mysql):
        self.mysql = mysql
        self.cleaner = TextCleaner()
        self.total = 0

    def fetch_list(self, url, category):
        if self.total >= MAX_TOTAL_NEWS:
            return

        print(f"\n==================== 开始爬取：{category} ====================")
        time.sleep(random.uniform(*DELAY))

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = "utf-8"
            sel = Selector(text=resp.text)
        except Exception as e:
            print(f"❌ 请求失败：{e}")
            return

        # 【关键修复】新浪通用新闻提取规则（支持多板块）
        items = sel.css('a::attr(href)').getall()
        valid_urls = []
        
        # 定义多种新闻URL模式
        news_patterns = [
            'doc-',           # 标准新闻文档
            '/i-',            # 短链接新闻
            '/c/',            # 汽车板块
            '/j/',            # 健康板块
            '/edu/',          # 教育板块
            '/travel/',       # 旅游板块
            '/fashion/',      # 时尚板块
            '/ent/',          # 娱乐板块
            '/sports/',       # 体育板块
            '/tech/',         # 科技板块
            '/finance/',      # 财经板块
        ]
        
        for u in items:
            if not u or 'sina.com.cn' not in u:
                continue
            # 检查是否匹配任何新闻模式
            is_news_url = any(pattern in u for pattern in news_patterns)
            if is_news_url and u not in valid_urls:
                valid_urls.append(u)

        if not valid_urls:
            print(f"⚠️ 未找到新闻链接，页面URL: {url}")
            print(f"   尝试提取的链接数: {len(items)}")
            if items:
                print(f"   前5个链接示例: {items[:5]}")
            return

        count = 0
        for news_url in valid_urls:
            if self.total >= MAX_TOTAL_NEWS or count >= MAX_CATEGORY_NEWS:
                break

            time.sleep(random.uniform(*DELAY))
            self.fetch_detail(news_url, category)
            count += 1

    def fetch_detail(self, url, category):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = "utf-8"
            sel = Selector(text=resp.text)
        except:
            return

        # 通用标题/正文提取（全网新浪文章通用）
        title = sel.css('h1::text').get() or sel.css('.main-title::text').get() or ''
        ps = sel.css('p::text').getall()
        content_raw = '\n'.join([p.strip() for p in ps if len(p.strip()) > 5])
        content_clean = self.cleaner.clean_text(content_raw)
        date_str = sel.css('.date::text').get() or sel.css('#pub_date::text').get() or ''
        publish_date = self.cleaner.parse_date(date_str)

        if not title or len(content_clean) < 30:
            return

        data = {
            "title": title.strip(),
            "content": content_clean,
            "source": "新浪新闻",
            "url": url,
            "category": category,
            "publish_date": publish_date,
            "read_count": 0,
            "comment_count": 0,
            "like_count": 0,
            "raw_content": content_raw
        }

        if self.mysql.insert_news(data):
            self.total += 1

# -------------------------- 主函数 --------------------------
def main():
    mysql = NewsMySQL()
    spider = SinaSpider(mysql)

    # 热门板块配置
    categories = [
        ("https://news.sina.com.cn/", "综合新闻"),
        ("https://finance.sina.com.cn/", "财经"),
        ("https://tech.sina.com.cn/", "科技"),
        ("https://sports.sina.com.cn/", "体育"),
        ("https://ent.sina.com.cn/", "娱乐"),
        ("https://edu.sina.com.cn/", "教育"),
        ("https://health.sina.com.cn/", "健康"),
        ("https://fashion.sina.com.cn/", "时尚"),
    ]
    
    print(f"开始爬取 {len(categories)} 个热门板块...")
    for url, category in categories:
        print(f"\n正在爬取板块: {category}")
        spider.fetch_list(url, category)

    mysql.close()
    print(f"\n🎉 爬取完成！总计入库：{spider.total} 条")

if __name__ == "__main__":
    main()