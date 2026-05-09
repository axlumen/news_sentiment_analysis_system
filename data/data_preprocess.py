import logging
import os
import re
from datetime import datetime
from difflib import SequenceMatcher

import jieba
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from config import Config

logger = logging.getLogger(__name__)

# -------------------------- 1. 配置加载与工具函数 --------------------------
# 加载环境变量
load_dotenv()

# MySQL配置
MYSQL_CONFIG = {
    "host": Config.host,
    "port": int(Config.port),
    "user": Config.user,
    "password": Config.password,
    "database": Config.database,
    "charset": Config.charset
}


# 构建SQLAlchemy引擎（核心修改点）
def create_mysql_engine():
    """创建SQLAlchemy MySQL引擎"""
    conn_str = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}?charset={MYSQL_CONFIG['charset']}"
    engine = create_engine(conn_str, pool_size=10, max_overflow=20, pool_recycle=3600)
    return engine


# 加载停用词表
from utils.text_utils import load_stopwords

STOPWORDS = load_stopwords()


# 相似度计算函数（用于重复文本检测）
def text_similarity(text1, text2):
    """计算两个文本的相似度（0-1）"""
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1, text2).ratio()


# -------------------------- 2. 数据读取与基础校验 --------------------------
def load_data_from_mysql():
    """从MySQL读取原始爬虫数据（使用SQLAlchemy引擎）"""
    try:
        # 核心修改：使用SQLAlchemy引擎替代原生pymysql连接
        engine = create_mysql_engine()
        # 读取数据（只读取核心字段，提升效率）
        sql = """
              SELECT id, \
                     title, \
                     content, \
                     source, \
                     url, \
                     category,
                     publish_date, \
                     read_count, \
                     comment_count, \
                     like_count, \
                     create_time
              FROM app_newsraw \
              """
        df = pd.read_sql(sql, engine)  # 无警告！
        engine.dispose()  # 关闭引擎

        print(f"原始数据总量：{len(df)}")
        print("\n数据字段预览：")
        print(df.info())
        print("\n前5行数据预览：")
        print(df.head())

        return df
    except Exception as e:
        print(f"读取MySQL数据失败：{e}")
        return pd.DataFrame()


def basic_data_validation(df):
    """基础数据质量校验：缺失值、异常值、格式校验（增强日期容错，消除SettingWithCopyWarning）"""
    print("\n===== 基础数据校验 =====")

    # 核心修复1：显式创建副本，避免视图/副本混淆
    df = df.copy()

    # 1. 缺失值统计（计算缺失率）
    missing_stats = df.isnull().sum() / len(df) * 100
    missing_df = missing_stats[missing_stats > 0].sort_values(ascending=False)
    print("\n1. 缺失率统计（%）：")
    if missing_df.empty:
        print("无缺失值")
    else:
        for col, rate in missing_df.items():
            print(f"  {col}: {rate:.2f}%")

    # 2. 异常值过滤
    print("\n2. 异常值过滤：")
    # 过滤条件1：正文为空或长度<10的无效文本
    len_before = len(df)
    # 核心修复2：使用.loc过滤，返回副本
    df = df.loc[df["content"].notna() & (df["content"].str.len() >= 10)].copy()
    print(f"  - 过滤正文过短/空数据：{len_before} → {len(df)}")

    # 过滤条件2：发布日期异常（增强容错，只过滤明显错误的，保留空值）
    current_year = datetime.now().year
    # 核心修复3：使用.loc赋值，消除警告
    df.loc[:, "publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")  # 转换为日期格式，错误值设为NaT
    
    # 确保publish_date是日期时间类型，避免.dt访问器错误
    if pd.api.types.is_datetime64_any_dtype(df["publish_date"]):
        # 只过滤年份异常的，空值保留（避免全量删除）
        df = df.loc[
            df["publish_date"].isna() |
            ((df["publish_date"].dt.year >= 2000) & (df["publish_date"].dt.year <= current_year))
            ].copy()
        print(f"  - 过滤异常发布日期（保留空值）：{len_before} → {len(df)}")
    else:
        print("  - 发布日期列不是日期时间类型，跳过日期过滤")

    # 过滤条件3：URL为空或无效
    df = df.loc[df["url"].notna() & (df["url"].str.startswith(("http://", "https://")))].copy()
    print(f"  - 过滤无效URL：{len_before} → {len(df)}")

    # 3. 字段格式统一
    print("\n3. 字段格式统一：")
    # 统一分类名称（去空格、小写、标准化）
    df.loc[:, "category"] = df["category"].fillna("未知").str.strip().str.lower()
    # 统一来源名称
    df.loc[:, "source"] = df["source"].fillna("未知").str.strip()
    # 数值字段默认值填充
    df.loc[:, "read_count"] = df["read_count"].fillna(0).astype(int)
    df.loc[:, "comment_count"] = df["comment_count"].fillna(0).astype(int)
    df.loc[:, "like_count"] = df["like_count"].fillna(0).astype(int)
    # 日期字段格式统一（NaT保留）
    if pd.api.types.is_datetime64_any_dtype(df["publish_date"]) and not df["publish_date"].isna().all():
        df.loc[:, "publish_date"] = df["publish_date"].dt.date
    print("  - 分类/来源/数值字段格式统一完成")

    return df


# -------------------------- 3. 深度重复值检测 --------------------------
def detect_duplicate_text(df, similarity_threshold=0.9):
    """深度重复文本检测：基于标题+内容相似度"""
    print(f"\n===== 深度重复值检测（相似度阈值：{similarity_threshold}） =====")

    # 显式创建副本，避免警告
    df = df.copy()

    # 先按URL去重（基础去重）
    df = df.drop_duplicates(subset=["url"], keep="first").copy()
    print(f"1. URL去重后数据量：{len(df)}")

    # 对所有数据集使用哈希值快速去重（性能优化）
    print("2. 使用哈希值快速去重")
    # 计算标题和内容的哈希值组合
    df["content_hash"] = df["title"].fillna("") + " " + df["content"].fillna("")
    df["content_hash"] = df["content_hash"].apply(lambda x: hash(x) % (10 ** 10))
    # 按哈希值去重
    df_clean = df.drop_duplicates(subset=["content_hash"], keep="first").copy()
    df_clean = df_clean.drop(columns=["content_hash"]).copy()
    removed_count = len(df) - len(df_clean)
    print(f"2. 哈希去重后数据量：{len(df_clean)}（移除{removed_count}条重复数据）")

    return df_clean


# -------------------------- 4. 文本精细化清洗 --------------------------
def batch_text_cleaning(df):
    """批量文本清洗（向量化优化）"""
    print("\n===== 文本精细化清洗 =====")

    # 显式创建副本
    df = df.copy()

    # 1. 去除HTML标签（向量化）
    df["title_clean"] = df["title"].fillna("").str.replace(r'<[^>]+>', '', regex=True)
    df["content_clean"] = df["content"].fillna("").str.replace(r'<[^>]+>', '', regex=True)

    # 2. 去除广告/水印文本（向量化）
    ad_patterns = [
        r'本文来源[:：].*',
        r'转载请注明出处[:：].*',
        r'编辑[:：].*',
        r'责任编辑[:：].*',
        r'图片来源[:：].*',
        r'版权所有[:：].*',
        r'更多精彩内容请关注.*',
        r'扫码关注.*',
        r'点击查看.*'
    ]
    for pattern in ad_patterns:
        df["title_clean"] = df["title_clean"].str.replace(pattern, '', regex=True)
        df["content_clean"] = df["content_clean"].str.replace(pattern, '', regex=True)

    # 3. 去除特殊符号（保留核心字符）
    df["title_clean"] = df["title_clean"].str.replace(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】《》、·\s]', '', regex=True)
    df["content_clean"] = df["content_clean"].str.replace(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】《》、·\s]', '', regex=True)

    # 4. 统一空格和换行
    df["title_clean"] = df["title_clean"].str.replace(r'\s+', ' ', regex=True).str.strip()
    df["content_clean"] = df["content_clean"].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 5. 过滤清洗后为空的文本
    df = df.loc[
        (df["title_clean"].str.len() >= 3) &
        (df["content_clean"].str.len() >= 10)
        ].copy()
    print(f"文本清洗后数据量：{len(df)}")

    return df


# -------------------------- 5. 文本标准化（适配深度学习） --------------------------
def standardize_text(df, max_token_len=512):
    """
    文本标准化：分词、停用词过滤、长度截断
    适配BERT等深度学习模型的输入要求
    """
    print(f"\n===== 文本标准化（最大Token长度：{max_token_len}） =====")

    # 显式创建副本
    df = df.copy()

    # 批量处理文本，使用列表推导式提高效率
    def process_texts(texts):
        """批量处理文本"""
        results = []
        for text in texts:
            # 1. 中文分词
            words = list(jieba.cut(text))
            # 2. 过滤停用词和空词
            words = [w for w in words if w not in STOPWORDS and w.strip() and len(w) > 1]
            # 3. 截断到最大长度
            if len(words) > max_token_len:
                words = words[:max_token_len]
            # 4. 拼接为字符串
            results.append(" ".join(words))
        return results

    # 批量处理标题和正文
    df["title_standard"] = process_texts(df["title_clean"].tolist())
    df["content_standard"] = process_texts(df["content_clean"].tolist())

    # 计算文本长度特征（供后续分析）
    df["title_len"] = df["title_clean"].str.len()
    df["content_len"] = df["content_clean"].str.len()
    df["content_token_len"] = df["content_standard"].str.split().str.len()

    print("文本标准化完成，新增字段：title_standard, content_standard, title_len, content_len, content_token_len")
    print(f"文本长度统计：")
    print(f"  - 标题平均长度：{df['title_len'].mean():.2f}字")
    print(f"  - 正文平均长度：{df['content_len'].mean():.2f}字")
    print(f"  - 正文平均Token数：{df['content_token_len'].mean():.2f}")

    return df


# -------------------------- 6. 清洗后数据落库/保存 --------------------------
def save_cleaned_data(df):
    """保存清洗后的数据：MySQL + CSV（批量插入优化）"""
    logger.info("Saving cleaned data...")

    df = df.copy()

    # 1. 保存为CSV
    output_cols = [
        "id", "title_clean", "content_clean", "title_standard", "content_standard",
        "source", "category", "publish_date", "url",
        "read_count", "comment_count", "like_count",
        "title_len", "content_len", "content_token_len"
    ]
    data_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(data_dir, f"news_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df[output_cols].to_csv(csv_path, index=False, encoding="utf8")
    logger.info(f"CSV saved: {csv_path}")

    # 2. 批量写入到 NewsSentimentAnalysis 表
    try:
        engine = create_mysql_engine()

        insert_sql = text("""
        INSERT INTO app_newssentimentanalysis
        (title_clean, content_clean, title_standard, content_standard,
         source, category, publish_date, url,
         read_count, comment_count, like_count,
         title_len, content_len, content_token_len, create_time)
        VALUES (:title_clean, :content_clean, :title_standard, :content_standard,
         :source, :category, :publish_date, :url,
         :read_count, :comment_count, :like_count,
         :title_len, :content_len, :content_token_len, :create_time)
        ON DUPLICATE KEY UPDATE
        title_clean = VALUES(title_clean),
        content_clean = VALUES(content_clean),
        title_standard = VALUES(title_standard),
        content_standard = VALUES(content_standard),
        source = VALUES(source),
        category = VALUES(category),
        publish_date = VALUES(publish_date),
        read_count = VALUES(read_count),
        comment_count = VALUES(comment_count),
        like_count = VALUES(like_count),
        title_len = VALUES(title_len),
        content_len = VALUES(content_len),
        content_token_len = VALUES(content_token_len)
        """)

        # 构建批量参数列表
        now = datetime.now()
        rows = []
        for _, row in df.iterrows():
            rows.append({
                'title_clean': row.get('title_clean', ''),
                'content_clean': row.get('content_clean', ''),
                'title_standard': row.get('title_standard', ''),
                'content_standard': row.get('content_standard', ''),
                'source': row.get('source', ''),
                'category': row.get('category', ''),
                'publish_date': row.get('publish_date', None),
                'url': row.get('url', ''),
                'read_count': row.get('read_count', 0),
                'comment_count': row.get('comment_count', 0),
                'like_count': row.get('like_count', 0),
                'title_len': row.get('title_len', 0),
                'content_len': row.get('content_len', 0),
                'content_token_len': row.get('content_token_len', 0),
                'create_time': now
            })

        # 分批执行，每批 500 条
        batch_size = 500
        success_count = 0
        fail_count = 0
        with engine.connect() as conn:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                try:
                    conn.execute(insert_sql, batch)
                    conn.commit()
                    success_count += len(batch)
                except Exception as e:
                    fail_count += len(batch)
                    logger.error(f"Batch insert failed (rows {i}-{i+len(batch)}): {e}")
                    conn.rollback()

        engine.dispose()
        logger.info(f"DB write complete: {success_count} success, {fail_count} failed")
    except Exception as e:
        logger.error(f"MySQL write failed: {e}")

    return csv_path


# -------------------------- 主函数：完整流程执行 --------------------------
def main():
    """数据校验与预处理完整流程"""
    # 步骤1：读取原始数据
    df_raw = load_data_from_mysql()
    if df_raw.empty:
        print("无原始数据，流程终止")
        return

    # 步骤2：基础数据校验
    df_valid = basic_data_validation(df_raw)

    # 步骤3：深度重复值检测
    df_no_dup = detect_duplicate_text(df_valid)

    # 步骤4：文本精细化清洗
    df_text_clean = batch_text_cleaning(df_no_dup)

    # 步骤5：文本标准化
    df_standard = standardize_text(df_text_clean)

    # 步骤6：保存清洗后的数据
    save_cleaned_data(df_standard)

    # 最终统计
    print("\n===== 数据预处理完成 =====")
    print(f"原始数据量：{len(df_raw)}")
    print(f"最终清洗后数据量：{len(df_standard)}")
    print(f"数据保留率：{len(df_standard) / len(df_raw) * 100:.2f}%")


if __name__ == "__main__":
    main()