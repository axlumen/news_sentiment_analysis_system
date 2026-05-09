import os
import re

import chardet
import jieba
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# 初始化Django环境
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'news_sentiment_analysis.settings')
django.setup()

from django.utils import timezone
from model.bilstm_attention import SentimentModel

# 导入模型
from app.models import NewsSentimentAnalysis


# -------------------------- 1. 核心修复：自动检测文件编码 + 兼容读取 --------------------------
def detect_file_encoding(file_path):
    """自动检测文件编码（解决gbk/utf-8乱码）"""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10240)  # 读取前10KB检测编码
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"📌 文件编码检测结果：{encoding}（置信度：{confidence:.2f}）")
        # 兼容常见中文编码
        if encoding is None:
            encoding = 'gbk'  # 默认兜底
        return encoding


def get_latest_cleaned_data(file_dir=None, prefix="news_cleaned_", suffix=".csv"):
    if file_dir is None:
        file_dir = _DATA_DIR
    """自动查找最新清洗数据文件"""
    cleaned_files = []
    for file in os.listdir(file_dir):
        if file.startswith(prefix) and file.endswith(suffix):
            time_match = re.search(r'(\d{8}_\d{6})', file)
            if time_match:
                time_str = time_match.group(1).replace("_", "")
                cleaned_files.append((file, int(time_str)))

    if not cleaned_files:
        raise FileNotFoundError(f"未找到符合规则的清洗数据文件（前缀：{prefix}，后缀：{suffix}）")

    cleaned_files.sort(key=lambda x: x[1], reverse=True)
    latest_file = cleaned_files[0][0]
    latest_file_path = os.path.join(file_dir, latest_file)
    print(f"✅ 自动识别到最新清洗数据文件：{latest_file_path}")
    return latest_file_path


def read_csv_safe(file_path):
    """安全读取CSV：自动适配编码，处理解码错误"""
    # 步骤1：检测编码
    encoding = detect_file_encoding(file_path)

    # 步骤2：多编码尝试读取
    encodings = [encoding, 'utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(
                file_path,
                encoding=enc,
                on_bad_lines='skip'  # 跳过错误行（替代error_bad_lines）
            )
            print(f"✅ 成功读取文件（编码：{enc}）")
            return df
        except Exception as e:
            print(f"⚠️ 编码{enc}读取失败：{e}")
            continue

    # 终极兜底：二进制模式读取后手动解码
    try:
        with open(file_path, 'rb') as f:
            content = f.read().decode('gbk', errors='ignore')  # 忽略解码错误
        from io import StringIO
        df = pd.read_csv(StringIO(content))
        print(f"✅ 兜底模式读取成功（忽略错误字符）")
        return df
    except Exception as e:
        raise RuntimeError(f"❌ 所有编码读取失败：{e}")


# -------------------------- 2. 核心配置 --------------------------
_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_DATA_DIR)
MODEL_PATH = os.path.join(_BASE_DIR, "model", "checkpoints", "best_model.pth")
VOCAB_PATH = os.path.join(_BASE_DIR, "model", "dataset", "processed", "vocab.pkl")
OUTPUT_PATH = os.path.join(_DATA_DIR, "news_sentiment_pred_latest.csv")

# 模型参数（与训练时一致）
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
MAX_LENGTH = 512
NUM_CLASSES = 3  # 三分类：0=负向, 1=中性, 2=正向
DROPOUT_RATE = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")

# 标签映射（与测试数据保持一致）
# 测试数据标签：0=负向, 1=中性, 2=正向
LABEL_MAP_REVERSE = {0: "negative", 1: "neutral", 2: "positive"}


# 加载停用词
def load_stopwords(file_path=None):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords.txt")
    try:
        with open(file_path, "r", encoding="utf8") as f:
            return set([line.strip() for line in f.readlines() if line.strip()])
    except FileNotFoundError:
        return {"的", "了", "在", "是", "我", "你", "他"}
    except UnicodeDecodeError:
        # 停用词文件编码兼容
        with open(file_path, "r", encoding="gbk") as f:
            return set([line.strip() for line in f.readlines() if line.strip()])


STOPWORDS = load_stopwords()


# -------------------------- 3. 文本预处理 & 预测器 --------------------------
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)  # 处理非字符串类型
    words = jieba.lcut(text)
    words = [w for w in words if w not in STOPWORDS and w.strip() and len(w) > 1]
    if len(words) > MAX_LENGTH:
        words = words[:MAX_LENGTH]
    else:
        words += ["<PAD>"] * (MAX_LENGTH - len(words))
    return words


def text_to_indices(words, vocab):
    return np.array([vocab.get(w, vocab["<UNK>"]) for w in words])


class SentimentPredictor:
    def __init__(self, model_path, vocab_path):
        # 词表加载（vocab.pkl是二进制文件，需要用rb模式读取）
        import pickle
        try:
            with open(vocab_path, "rb") as f:
                vocab_obj = pickle.load(f)
            
            # 兼容两种保存格式：直接 vocab 或 包含 vocab 字段的 dict
            if isinstance(vocab_obj, dict) and "vocab" in vocab_obj:
                self.vocab = vocab_obj["vocab"]
            else:
                self.vocab = vocab_obj
        except Exception as e:
            print(f"词表加载失败：{e}")
            # 使用默认词表（仅用于测试）
            self.vocab = {"<PAD>": 0, "<UNK>": 1}

        self.vocab_size = len(self.vocab)
        print(f"词表加载成功，大小：{self.vocab_size}")

        self.model = SentimentModel(
            vocab_size=self.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=NUM_CLASSES,
            n_layers=NUM_LAYERS,
            dropout=DROPOUT_RATE
        ).to(DEVICE)

        # 加载模型权重（兼容CPU/GPU和不同保存格式）
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # 检查checkpoint的格式
        if isinstance(checkpoint, dict):
            # 尝试不同的键名
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                # 尝试直接加载（如果checkpoint本身就是state_dict）
                try:
                    self.model.load_state_dict(checkpoint)
                except Exception as e:
                    raise RuntimeError(f"无法加载模型权重：{e}")
        else:
            # 直接加载
            self.model.load_state_dict(checkpoint)
            
        self.model.eval()
        print(f"模型加载成功：{model_path}")

    def predict_batch(self, texts, batch_size=8):
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="批量预测"):
            batch_texts = texts[i:i + batch_size]
            batch_indices = []

            # 批量预处理（增加异常处理）
            for text in batch_texts:
                try:
                    words = preprocess_text(text)
                    indices = text_to_indices(words, self.vocab)
                    batch_indices.append(indices)
                except Exception as e:
                    print(f"⚠️ 文本预处理失败：{text[:50]}... 错误：{e}")
                    batch_indices.append(np.zeros(MAX_LENGTH, dtype=int))  # 空值兜底

            # 模型推理（优化：先转换为numpy数组，再创建张量）
            batch_indices_np = np.array(batch_indices)
            x = torch.tensor(batch_indices_np, dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                logits, _ = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            # 解析结果
            # 概率索引：0=negative, 1=neutral, 2=positive
            for j, prob in enumerate(probs):
                pred_label_idx = np.argmax(prob)
                pred_label = LABEL_MAP_REVERSE[pred_label_idx]
                results.append({
                    "sentiment": pred_label,
                    "sentiment_score": round(prob[pred_label_idx], 4),
                    "negative_prob": round(prob[0], 4),
                    "neutral_prob": round(prob[1], 4),
                    "positive_prob": round(prob[2], 4)
                })
        return results
    
    def predict_single(self, text):
        """单个文本的情感预测"""
        try:
            # 文本预处理
            words = preprocess_text(text)
            indices = text_to_indices(words, self.vocab)
            
            # 转换为张量
            x = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
            
            # 模型推理
            with torch.no_grad():
                logits, _ = self.model(x)
                prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # 解析结果
            # 概率索引：0=negative, 1=neutral, 2=positive
            pred_label_idx = np.argmax(prob)
            pred_label = LABEL_MAP_REVERSE[pred_label_idx]
            
            # 确保返回的是 Python 原生类型，而不是 numpy 类型
            return {
                "sentiment": pred_label,
                "sentiment_score": float(round(prob[pred_label_idx], 4)),
                "negative_prob": float(round(prob[0], 4)),
                "neutral_prob": float(round(prob[1], 4)),
                "positive_prob": float(round(prob[2], 4))
            }
        except Exception as e:
            print(f"⚠️ 单个文本预测失败：{text[:50]}... 错误：{e}")
            # 返回默认值
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.3333,
                "negative_prob": 0.3333,
                "neutral_prob": 0.3334,
                "positive_prob": 0.3333
            }


# -------------------------- 5. 主预测流程 --------------------------
def process_existing_records():
    """处理数据库中已有的但没有情感分析结果的记录"""
    print("\n处理数据库中已有的未分析记录...")
    
    # 查询数据库中没有情感分析结果的记录
    unanalyzed_news = NewsSentimentAnalysis.objects.filter(
        sentiment__isnull=True
    )
    
    if unanalyzed_news.count() == 0:
        print("✅ 数据库中没有未分析的记录")
        return 0
    
    print(f"发现 {unanalyzed_news.count()} 条未分析的记录")
    
    # 初始化预测器
    try:
        predictor = SentimentPredictor(MODEL_PATH, VOCAB_PATH)
    except Exception as e:
        print(f"❌ 预测器初始化失败：{e}")
        return 0
    
    # 处理未分析的记录
    processed_count = 0
    for news in unanalyzed_news:
        try:
            # 使用content_standard或content_clean作为预测文本
            text = news.content_standard or news.content_clean or news.title_clean
            if not text:
                continue
            
            # 预测情感
            result = predictor.predict_single(text)
            
            # 更新数据库
            news.sentiment = result['sentiment']
            news.sentiment_score = result['sentiment_score']
            news.save()
            
            processed_count += 1
        except Exception as e:
            print(f"处理记录失败 (ID: {news.id}): {e}")
    
    print(f"✅ 处理完成，共更新 {processed_count} 条记录")
    return processed_count

def main():
    # 优先处理数据库中已有的未分析记录
    print("\n优先处理数据库中已有的未分析记录...")
    processed_count = process_existing_records()
    
    # 如果没有处理到记录，再尝试从CSV文件读取
    if processed_count == 0:
        print("\n未找到未分析的记录，尝试从CSV文件读取数据...")
        
        # 1. 自动获取最新清洗数据文件
        try:
            DATA_PATH = get_latest_cleaned_data()
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return

        # 2. 安全加载数据（核心修复：解决编码错误）
        try:
            df = read_csv_safe(DATA_PATH)
        except RuntimeError as e:
            print(f"❌ 数据读取失败：{e}")
            return
        print(f"待预测数据量：{len(df)}")

        # 3. 处理空值
        if "content_standard" not in df.columns:
            print("❌ 数据缺少content_standard列")
            return
        texts = df["content_standard"].fillna("").tolist()

        # 4. 初始化预测器
        try:
            predictor = SentimentPredictor(MODEL_PATH, VOCAB_PATH)
        except Exception as e:
            print(f"❌ 预测器初始化失败：{e}")
            return

        # 5. 批量预测
        batch_size = 8 if torch.cuda.is_available() else 2
        pred_results = predictor.predict_batch(texts, batch_size=batch_size)

        # 6. 合并结果并保存（指定编码为utf-8，避免写入乱码）
        pred_df = pd.DataFrame(pred_results)
        final_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
        final_df.to_csv(
            OUTPUT_PATH,
            index=False,
            encoding="utf-8-sig",  # utf-8-sig解决Excel打开乱码
            errors="ignore"
        )
        print(f"\n✅ 预测结果已保存：{OUTPUT_PATH}")

        # 7. 保存预测结果到数据库
        print("\n正在保存预测结果到数据库...")
        saved_count = 0
        for idx, row in final_df.iterrows():
            # 检查是否已存在（基于url）
            url = row.get('url', '')
            if not url:
                continue

            try:
                # 确保标题不为空，优先从title_standard获取，其次从title_clean获取
                title = row.get('title_clean', '') or row.get('title', '')
                if not title:
                    # 如果标题为空，尝试从内容中提取前50个字符作为标题
                    content = row.get('content_standard', '') or row.get('content', '')
                    if content:
                        title = content[:50] + '...' if len(content) > 50 else content
                    else:
                        title = '无标题新闻'
                
                # 尝试获取发布日期
                publish_date = None
                if 'publish_date' in row:
                    try:
                        # 转换为日期时间对象
                        date_obj = pd.to_datetime(row['publish_date'])
                        # 检查是否为 NaT
                        if pd.isna(date_obj):
                            publish_date = None
                        else:
                            # 转换为Python datetime对象
                            if hasattr(date_obj, 'to_pydatetime'):
                                publish_date = date_obj.to_pydatetime()
                                # 确保时区信息存在
                                if publish_date.tzinfo is None:
                                    # 添加默认时区（使用Django设置的时区）
                                    publish_date = timezone.make_aware(publish_date)
                    except:
                        pass
                
                news, created = NewsSentimentAnalysis.objects.get_or_create(
                    url=url,
                    defaults={
                        'title_clean': title,
                        'content_clean': row.get('content_clean', ''),
                        'title_standard': row.get('title_standard', ''),
                        'content_standard': row.get('content_standard', ''),
                        'source': row.get('source', ''),
                        'category': row.get('category', ''),
                        'publish_date': publish_date,
                        'sentiment': row.get('sentiment', ''),
                        'sentiment_score': row.get('sentiment_score', 0.0),
                    }
                )
                
                # 更新所有字段
                news.title_clean = title
                news.content_clean = row.get('content_clean', '')
                news.title_standard = row.get('title_standard', '')
                news.content_standard = row.get('content_standard', '')
                news.source = row.get('source', '')
                news.category = row.get('category', '')
                news.publish_date = publish_date
                news.sentiment = row.get('sentiment', '')
                news.sentiment_score = row.get('sentiment_score', 0.0)
                news.save()
                
                saved_count += 1
            except Exception as e:
                print(f"保存记录失败（URL: {url}）: {e}")
        
        print(f"\n✅ 预测结果已保存到数据库，共保存 {saved_count} 条记录")
    
    # 更新统计模型
    print("\n正在更新统计模型...")
    try:
        from data.news_analysis import analyze_and_save
        analyze_and_save()
        print("✅ 统计模型更新成功！")
    except Exception as e:
        print(f"⚠️ 统计模型更新失败：{e}")

    # 示例：单条预测
    test_text = "今日A股三大指数集体上涨，新能源板块领涨，市场情绪积极。"
    try:
        # 重新初始化预测器（可能没有初始化）
        if 'predictor' not in locals():
            predictor = SentimentPredictor(MODEL_PATH, VOCAB_PATH)
        
        test_words = preprocess_text(test_text)
        test_indices = text_to_indices(test_words, predictor.vocab)
        test_x = torch.tensor(test_indices, dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, _ = predictor.model(test_x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = LABEL_MAP_REVERSE[np.argmax(prob)]
        print(f"\n📌 单条预测示例：")
        print(f"文本：{test_text}")
        print(f"预测情感：{pred_label}（置信度：{prob[np.argmax(prob)]:.4f}）")
    except Exception as e:
        print(f"⚠️ 单条预测失败：{e}")


if __name__ == "__main__":
    main()