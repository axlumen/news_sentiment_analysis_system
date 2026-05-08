import pandas as pd
import pymysql
from datetime import datetime
import jieba
from collections import Counter
from sqlalchemy import create_engine

from config import Config

# ========== 数据库配置 ==========
# MySQL配置
MYSQL_CONFIG = {
    "host": Config.host,
    "port": int(Config.port),
    "user": Config.user,
    "password": Config.password,
    "database": Config.database,
    "charset": Config.charset
}

engine = create_engine(
    f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}?charset={MYSQL_CONFIG['charset']}"
)

RAW_NEWS_TABLE = "app_newssentimentanalysis"

# ========== 工具函数 ==========
def get_conn():
    return pymysql.connect(**MYSQL_CONFIG)

# ========== 创建正确的表 ==========
def create_stat_tables():
    conn = get_conn()
    c = conn.cursor()

    # 1. 整体情感
    c.execute("""
    CREATE TABLE IF NOT EXISTS overall_sentiment (
        id INT PRIMARY KEY AUTO_INCREMENT,
        total_news INT,
        positive_count INT,
        neutral_count INT,
        negative_count INT,
        positive_pct FLOAT,
        neutral_pct FLOAT,
        negative_pct FLOAT,
        avg_sentiment_score FLOAT,
        create_time DATETIME DEFAULT NOW()
    )
    """)

    # 2. 来源情感
    c.execute("""
    CREATE TABLE IF NOT EXISTS source_sentiment (
        id INT PRIMARY KEY AUTO_INCREMENT,
        source VARCHAR(100),
        total_count INT,
        positive_count INT,
        neutral_count INT,
        negative_count INT,
        avg_sentiment_score FLOAT
    )
    """)

    # 3. 分类情感
    c.execute("""
    CREATE TABLE IF NOT EXISTS category_sentiment (
        id INT PRIMARY KEY AUTO_INCREMENT,
        category VARCHAR(100),
        total_count INT,
        positive_count INT,
        neutral_count INT,
        negative_count INT,
        avg_sentiment_score FLOAT
    )
    """)

    # 4. 日期情感
    c.execute("""
    CREATE TABLE IF NOT EXISTS date_sentiment (
        id INT PRIMARY KEY AUTO_INCREMENT,
        date_str VARCHAR(20),
        total_count INT,
        positive_count INT,
        neutral_count INT,
        negative_count INT,
        avg_sentiment_score FLOAT
    )
    """)

    # 5. 得分分布
    c.execute("""
    CREATE TABLE IF NOT EXISTS score_distribution (
        id INT PRIMARY KEY AUTO_INCREMENT,
        score_range VARCHAR(30),
        count INT
    )
    """)

    # 6. 关键词
    c.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_keywords (
        id INT PRIMARY KEY AUTO_INCREMENT,
        keyword VARCHAR(50),
        frequency INT,
        sentiment VARCHAR(20)
    )
    """)

    conn.commit()
    c.close()
    conn.close()
    print("✅ 所有统计表已创建（带完整字段）")

# ========== 全量统计入库 ==========
def analyze_and_save():
    print("开始统计分析...")
    # 先查询总记录数
    total_query = f"SELECT COUNT(*) FROM {RAW_NEWS_TABLE}"
    total_df = pd.read_sql(total_query, engine)
    total_records = total_df.iloc[0, 0]
    print(f"总记录数: {total_records}")
    
    # 查询有情感数据的记录数
    sql = f"""
    SELECT id,title_clean,content_clean,source,category,publish_date,sentiment,sentiment_score
    FROM {RAW_NEWS_TABLE}
    WHERE sentiment IS NOT NULL AND sentiment_score IS NOT NULL
    """
    df = pd.read_sql(sql, engine)

    print(f"有情感数据的记录数: {len(df)}")
    
    if len(df) == 0:
        print("❌ 无数据")
        return

    total = len(df)
    conn = get_conn()
    c = conn.cursor()

    # 清空旧表
    print("清空旧统计数据...")
    c.execute("TRUNCATE overall_sentiment")
    c.execute("TRUNCATE source_sentiment")
    c.execute("TRUNCATE category_sentiment")
    c.execute("TRUNCATE date_sentiment")
    c.execute("TRUNCATE score_distribution")
    c.execute("TRUNCATE sentiment_keywords")

    # ============== 1. 整体 ==============
    pos = len(df[df['sentiment'] == 'positive'])
    neu = len(df[df['sentiment'] == 'neutral'])
    neg = len(df[df['sentiment'] == 'negative'])
    pos_pct = round(pos/total*100,2)
    neu_pct = round(neu/total*100,2)
    neg_pct = round(neg/total*100,2)
    avg_score = round(df['sentiment_score'].mean(),3)

    c.execute("""
    INSERT INTO overall_sentiment VALUES
    (null,%s,%s,%s,%s,%s,%s,%s,%s,now())
    """,(total,pos,neu,neg,pos_pct,neu_pct,neg_pct,avg_score))

    # ============== 2. 来源 ==============
    source_df = df.groupby('source')['sentiment'].value_counts().unstack(fill_value=0)
    source_avg = df.groupby('source')['sentiment_score'].mean().round(3)
    for s in source_df.index:
        tc = int(source_df.loc[s].sum())
        p = int(source_df.loc[s].get('positive',0))
        n = int(source_df.loc[s].get('neutral',0))
        ng = int(source_df.loc[s].get('negative',0))
        avg = float(source_avg.loc[s])
        c.execute("INSERT INTO source_sentiment VALUES (null,%s,%s,%s,%s,%s,%s)",
                  (s,tc,p,n,ng,avg))

    # ============== 3. 分类 ==============
    cate_df = df.groupby('category')['sentiment'].value_counts().unstack(fill_value=0)
    cate_avg = df.groupby('category')['sentiment_score'].mean().round(3)
    for cname in cate_df.index:
        tc = int(cate_df.loc[cname].sum())
        p = int(cate_df.loc[cname].get('positive',0))
        n = int(cate_df.loc[cname].get('neutral',0))
        ng = int(cate_df.loc[cname].get('negative',0))
        avg = float(cate_avg.loc[cname])
        c.execute("INSERT INTO category_sentiment VALUES (null,%s,%s,%s,%s,%s,%s)",
                  (cname,tc,p,n,ng,avg))

    # ============== 4. 日期 ==============
    df['date_str'] = pd.to_datetime(df['publish_date'],errors='coerce').dt.strftime('%Y-%m-%d')
    date_df = df.dropna(subset=['date_str']).groupby('date_str')['sentiment'].value_counts().unstack(fill_value=0)
    date_avg = df.groupby('date_str')['sentiment_score'].mean().round(3)
    for dt in date_df.index:
        tc = int(date_df.loc[dt].sum())
        p = int(date_df.loc[dt].get('positive',0))
        n = int(date_df.loc[dt].get('neutral',0))
        ng = int(date_df.loc[dt].get('negative',0))
        avg = float(date_avg.loc[dt])
        c.execute("INSERT INTO date_sentiment VALUES (null,%s,%s,%s,%s,%s,%s)",
                  (dt,tc,p,n,ng,avg))

    # ============== 5. 得分分布 ==============
    bins = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    labels = ["-1.0~-0.8","-0.8~-0.6","-0.6~-0.4","-0.4~-0.2","-0.2~0",
              "0~0.2","0.2~0.4","0.4~0.6","0.6~0.8","0.8~1.0"]
    df['score_range'] = pd.cut(df['sentiment_score'],bins=bins,labels=labels,include_lowest=True)
    for lbl,cnt in df['score_range'].value_counts().sort_index().items():
        c.execute("INSERT INTO score_distribution VALUES (null,%s,%s)",(str(lbl),int(cnt)))

    # ============== 6. 关键词 ==============
    stop_words = {"的","了","是","在","和","就","都","一个","我","你","他"}
    words = []
    for txt in df['content_clean'].dropna():
        words += [w for w in jieba.lcut(str(txt)[:5000]) if len(w)>=2 and w not in stop_words]
    for w,f in Counter(words).most_common(50):
        c.execute("INSERT INTO sentiment_keywords VALUES (null,%s,%s,'all')",(w,f))

    conn.commit()
    c.close()
    conn.close()
    print(f"✅ 统计完成！共 {total} 条数据，所有表字段完整")

if __name__ == '__main__':
    create_stat_tables()
    analyze_and_save()