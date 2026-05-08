import os
import sys
import re
import gc
import json
import multiprocessing as mp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pandarallel import pandarallel
import opencc  # 用于中文繁简体转换

"""
init_data.py

本脚本负责完成原始新闻情感分析数据集的预处理与统计分析，主要包括：
1. 原始 CSV 读取与基础字段规范化（列名对齐）。
2. 文本内容清洗：HTML 标签去除、繁体转简体、空白规范化、长度过滤等。
3. 评分字段标准化与情感标签生成（1–2 分视为负向，3 分视为中性，4–5 分视为正向）。
4. 文本去重，防止相同新闻出现在多个划分中导致信息泄露。
5. 可选的类别平衡（下采样）操作（默认关闭，仅保留真实分布）。
6. 文本长度分布统计与可视化（用于确定模型的最大序列长度 MAX_LEN）。
7. 按 70% / 10% / 20% 的比例进行 Train / Val / Test 分层划分。
8. 将关键统计信息和数据谱系信息写入 meta.json，便于论文复现和结果说明。

设计目标是保证实验数据处理过程完全可复现，并为论文的数据描述部分提供可直接引用的数字与图表。
"""

# ============================
# 全局配置
# ============================

# 数据目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR

# 原始文件路径（位于 dataset/raw/）
RAW_DIR = os.path.join(DATA_DIR, "../dataset/raw")

# 预处理后的输出文件路径（位于 dataset/processed/）
PROCESSED_DIR = os.path.join(DATA_DIR, "../dataset/processed")
FIGURES_DIR = os.path.join(DATA_DIR, "../dataset/figures")

TRAIN_FILE = os.path.join(PROCESSED_DIR, "train.csv")
VAL_FILE = os.path.join(PROCESSED_DIR, "val.csv")
TEST_FILE = os.path.join(PROCESSED_DIR, "test.csv")
META_FILE = os.path.join(PROCESSED_DIR, "meta.json")

# 文本长度分布图（位于 dataset/figures/）
PLOT_FILE = os.path.join(FIGURES_DIR, "length_distribution.png")

# 文本过滤与随机性控制参数
MIN_LEN = 5            # 文本最小长度阈值：用于过滤极短且信息量有限的评论
SEED = 42              # 全局随机种子：保证数据划分和采样过程的可复现性
ENABLE_BALANCE = False # 是否启用类别平衡（下采样）；默认关闭，保留真实分布

# ============================
# 正则表达式与辅助工具初始化
# ============================

# HTML 标签匹配正则，用于去除富文本标签
RE_HTML = re.compile(r"<[^>]+>")

# 多余空白字符（含空格、制表符、换行等）匹配正则
RE_SPACE = re.compile(r"\s+")

# OpenCC 繁简体转换器：t2s 表示 Traditional Chinese → Simplified Chinese
CONVERTER = opencc.OpenCC("t2s")


class NpEncoder(json.JSONEncoder):
    """
    自定义 JSONEncoder，用于安全地序列化 numpy 类型（如 np.int64、np.float64、np.ndarray 等）。

    标准 json 库无法直接序列化 numpy 的数值类型，会抛出 TypeError。
    该类通过类型检查将 numpy 类型转换为原生 Python 类型（int、float、list）后再序列化。
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def clean_text_content(text: str) -> str:
    """
    文本清洗函数（Text Cleaning Pipeline）。

    该函数在训练阶段与推理阶段应保持一致，用于构建统一的文本空间。
    主要处理步骤：
        1. 去除 HTML 标签：剥离富文本标记，仅保留自然语言内容。
        2. 繁体转简体：将繁体中文统一转换为简体中文，避免同词多写造成的数据稀疏。
        3. 空白规整：将多余的空白字符（空格、换行、制表符等）压缩为单个空格，并去除首尾空白。

    参数：
        text (str): 原始新闻文本内容。

    返回：
        str: 清洗后的文本。如输入为非字符串或清洗后为空，返回空字符串 ""。
    """
    if not isinstance(text, str):
        return ""

    # 定义局部正则表达式以避免多进程问题
    import re
    RE_HTML = re.compile(r"<[^>]+>")
    RE_SPACE = re.compile(r"\s+")
    import opencc
    CONVERTER = opencc.OpenCC("t2s")

    # 去除 HTML 标签
    text = RE_HTML.sub("", text)

    # 繁体转简体（中文统一化处理）
    text = CONVERTER.convert(text)

    # 空白规范化
    text = RE_SPACE.sub(" ", text)

    # 去除首尾空白
    text = text.strip()

    if not text:
        return ""

    return text


def _check_and_fix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    列名标准化与字段抽取函数。

    该函数的目标是从原始 DataFrame 中提取出“新闻文本列 (text)”和“情感标签列 (rating)”，
    并对不规范的列名进行兼容性处理。

    处理逻辑：
        1. 对所有列名做小写与去空格处理，避免大小写或多余空白带来的匹配问题。
        2. 尝试使用常见列名进行语义对齐：
            - 'comment' / 'review' / 'content' / 'news' / 'sentence' → 'text'
            - 'star' / 'score' / 'sentiment' / 'label' → 'rating'
        3. 如果对齐后仍未包含 'text' 或 'rating'：
            - 尝试按照约定位置抽取：根据实际列数调整抽取策略。
            - 对抽取结果进行简单的类型合理性检查：
                * 文本列中字符串比例应不低于 50%。
                * 评分列中可成功转为数值的比例应不低于 50%。
            - 若检查不通过，则认为 CSV 结构与预期不符，直接报错退出。
        4. 如果 'rating' 列包含文本标签（如 '积极', '中性', '消极'），则将其转换为数值：
            - '积极' → 2 (正向)
            - '中性' → 1 (中性)
            - '消极' → 0 (负向)

    参数：
        df (pd.DataFrame): 原始读入的 DataFrame。

    返回：
        pd.DataFrame: 仅保留 'text' 和 'rating' 两列的新 DataFrame。
    """
    # 列名标准化
    df.columns = [c.lower().strip() for c in df.columns]

    # 常见列名映射
    rename_map = {
        "comment": "text",
        "review": "text",
        "content": "text",
        "news": "text",  # 添加 news 作为文本列
        "sentence": "text",  # 添加 sentence 作为文本列
        "star": "rating",
        "score": "rating",
        "label": "rating",  # 添加 label 作为评分列
    }
    df.rename(columns=rename_map, inplace=True)

    # 优先使用语义字段抽取
    if "rating" in df.columns and "text" in df.columns:
        df_result = df[["text", "rating"]].copy()
        
        # 检查 'rating' 列是否包含文本标签，如果是则转换为数值
        sample_ratings = df_result["rating"].dropna().head(10)  # 取前10个非空值作为样本
        if sample_ratings.dtype == "object":  # 如果是文本类型
            # 创建标签映射
            label_mapping = {
                "积极": 2, "positive": 2, "pos": 2,
                "中性": 1, "neutral": 1, "neu": 1,
                "消极": 0, "negative": 0, "neg": 0,
            }
            
            # 检查样本中是否有匹配的标签
            sample_values = sample_ratings.astype(str).str.lower().unique()
            if any(val in label_mapping for val in sample_values):
                print("检测到文本情感标签，正在转换为数值标签...")
                df_result["rating"] = df_result["rating"].astype(str).str.lower().map(label_mapping)
                # 处理未映射的值
                unmapped_mask = df_result["rating"].isna()
                if unmapped_mask.any():
                    print(f"警告：发现无法映射的情感标签: {df_result[unmapped_mask]['rating'].unique()}")
                    print("仅保留已成功映射的样本。")
                    df_result = df_result.dropna(subset=["rating"])
                
                # 转换为整数类型
                df_result["rating"] = df_result["rating"].astype(int)
                print("文本情感标签已成功转换为数值标签（0=消极, 1=中性, 2=积极）")
        
        return df_result

    # 回退方案：按固定列位置抽取，并做合理性检查
    print("警告：未识别到标准列名，尝试按位置提取字段。")
    print(f"    CSV 文件共有 {len(df.columns)} 列。")
    print(f"    列名: {df.columns.tolist()}")
    
    try:
        # 根据实际列数调整抽取策略
        if len(df.columns) >= 5:
            # 有5列或更多，使用原策略
            tmp = df.iloc[:, [4, 2]].copy()
            tmp.columns = ["text", "rating"]
            print("    尝试抽取：第 5 列为 text，第 3 列为 rating")
        elif len(df.columns) == 4:
            # 只有4列，尝试第4列为text，第2列为rating
            tmp = df.iloc[:, [3, 1]].copy()
            tmp.columns = ["text", "rating"]
            print("    尝试抽取：第 4 列为 text，第 2 列为 rating")
        elif len(df.columns) == 3:
            # 只有3列，尝试第3列为text，第1列为rating
            tmp = df.iloc[:, [2, 0]].copy()
            tmp.columns = ["text", "rating"]
            print("    尝试抽取：第 3 列为 text，第 1 列为 rating")
        elif len(df.columns) == 2:
            # 只有2列，假设第2列为text，第1列为rating
            tmp = df.iloc[:, [1, 0]].copy()
            tmp.columns = ["text", "rating"]
            print("    尝试抽取：第 2 列为 text，第 1 列为 rating")
        else:
            # 列数不足，无法抽取
            raise ValueError(f"CSV 文件只有 {len(df.columns)} 列，无法抽取 text 和 rating 字段。")

        # 文本列：检查“是否为字符串”的比例
        text_is_str_ratio = tmp["text"].map(lambda x: isinstance(x, str)).mean()

        # 评分列：检查“是否可转换为数值”的比例
        rating_is_num_ratio = pd.to_numeric(tmp["rating"], errors="coerce").notna().mean()

        print(f"    文本列为字符串比例: {text_is_str_ratio:.3f}")
        print(f"    评分列为数值比例:   {rating_is_num_ratio:.3f}")

        # 若任一比例过低，则认为列抽取可能错误，直接报错终止
        if text_is_str_ratio < 0.5 or rating_is_num_ratio < 0.5:
            raise ValueError("按位置抽列后的类型检测不通过，疑似列错位。")

        # 检查评分列是否包含文本标签，如果是则转换为数值
        sample_ratings = tmp["rating"].dropna().head(10)  # 取前10个非空值作为样本
        if sample_ratings.dtype == "object":  # 如果是文本类型
            # 创建标签映射
            label_mapping = {
                "积极": 2, "positive": 2, "pos": 2,
                "中性": 1, "neutral": 1, "neu": 1,
                "消极": 0, "negative": 0, "neg": 0,
            }
            
            # 检查样本中是否有匹配的标签
            sample_values = sample_ratings.astype(str).str.lower().unique()
            if any(val in label_mapping for val in sample_values):
                print("检测到文本情感标签，正在转换为数值标签...")
                tmp["rating"] = tmp["rating"].astype(str).str.lower().map(label_mapping)
                # 处理未映射的值
                unmapped_mask = tmp["rating"].isna()
                if unmapped_mask.any():
                    print(f"警告：发现无法映射的情感标签: {tmp[unmapped_mask]['rating'].unique()}")
                    print("仅保留已成功映射的样本。")
                    tmp = tmp.dropna(subset=["rating"])
                
                # 转换为整数类型
                tmp["rating"] = tmp["rating"].astype(int)
                print("文本情感标签已成功转换为数值标签（0=消极, 1=中性, 2=积极）")

        return tmp
    except Exception as e:
        print(f"错误：CSV 结构不符合预期，无法可靠抽取 text/rating 列。原因：{e}")
        print("\n请确保 CSV 文件包含以下列之一：")
        print("  - 文本列：text, comment, review, content, news, sentence")
        print("  - 评分列：rating, star, score, sentiment, label")
        print("\n或者确保 CSV 文件至少包含两列数据。")
        sys.exit(1)


def process_data():
    """
    数据预处理流程主函数。

    该函数按照如下顺序依次执行：
        1. 并行环境初始化。
        2. 原始 CSV 文件加载与字段规范化。
        3. 文本清洗与最小长度过滤。
        4. 评分字段数值化与合法范围约束。
        5. 生成三分类情感标签（评分=1-2 负向，3 中性，4-5 正向）。
        6. 文本内容去重。
        7. （可选）类别平衡下采样（默认关闭）。
        8. 文本长度分布统计与可视化。
        9. 按 70% / 10% / 20% 比例进行分层划分（stratified split）。
       10. 将关键统计信息与数据谱系信息写入 meta.json。
       11. 保存 train/val/test 三个子集为 CSV 文件。
    """
    print("[Step 1] 启动数据清洗与统计流程。")

    # ----------------------------------------------------
    # 0. 初始化并行加速环境（基于 pandarallel）
    # ----------------------------------------------------
    try:
        cpu_cnt = mp.cpu_count()
        num_workers = min(16, cpu_cnt)  # 最多使用 16 个进程，避免在小型设备上过载
        print(f"信息：初始化 pandarallel，使用 {num_workers}/{cpu_cnt} 个 CPU 核心。")
        pandarallel.initialize(progress_bar=True, nb_workers=num_workers, verbose=1)
    except Exception as e:
        print(f"警告：并行初始化失败（{e}），后续将回退到单进程模式。")

    # ----------------------------------------------------
    # 1. 读取原始数据文件
    # ----------------------------------------------------
    if not os.path.exists(RAW_DIR):
        print(f"错误：未找到原始数据目录 {RAW_DIR}，请检查路径配置。")
        sys.exit(1)

    print("信息：开始读取原始数据 CSV 文件。")
    
    # 定义文件标签映射
    file_label_map = {
        "neg.csv": 0,      # 消极
        "neutral.csv": 1,  # 中性
        "pos.csv": 2       # 积极
    }
    
    # 读取所有文件并合并
    dfs = []
    for filename, label in file_label_map.items():
        file_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(file_path):
            print(f"警告：未找到文件 {file_path}，跳过该文件。")
            continue
        
        try:
            df_file = pd.read_csv(file_path, on_bad_lines="skip", encoding="utf-8")
            # 检查文件是否有内容
            if len(df_file) == 0:
                print(f"警告：文件 {filename} 为空，跳过该文件。")
                continue
            
            # 处理文本列
            if 'text' in df_file.columns:
                text_col = 'text'
            elif 'content' in df_file.columns:
                text_col = 'content'
            elif 'comment' in df_file.columns:
                text_col = 'comment'
            elif 'review' in df_file.columns:
                text_col = 'review'
            elif len(df_file.columns) > 0:
                # 取第一列作为文本列
                text_col = df_file.columns[0]
            else:
                print(f"警告：文件 {filename} 没有列，跳过该文件。")
                continue
            
            # 构建新的 DataFrame
            df_processed = pd.DataFrame({
                'text': df_file[text_col],
                'rating': label  # 直接使用文件对应的标签
            })
            dfs.append(df_processed)
            print(f"信息：读取文件 {filename}，样本数: {len(df_processed):,}")
        except Exception as e:
            print(f"错误：读取文件 {filename} 失败：{e}")
            continue
    
    if not dfs:
        print("错误：没有成功读取任何数据文件，请检查文件格式。")
        sys.exit(1)
    
    # 合并所有数据
    df_raw = pd.concat(dfs, ignore_index=True)
    raw_count = len(df_raw)
    print(f"信息：原始样本总数: {raw_count:,}")

    # 由于我们已经手动构建了包含 text 和 rating 的 DataFrame，直接使用
    df = df_raw.copy()
    print(f"信息：字段对齐后样本数（text/rating 可用）: {len(df):,}")

    # 数据谱系信息，用于记录每一步筛选后的样本规模变化
    lineage = {
        "raw_count": int(raw_count),
        "after_column_normalization": int(len(df)),
        "after_clean_and_minlen": None,
        "length_filter_drop": None,
        "after_rating_valid": None,
        "after_drop_neutral_3": None,
        "neutral_3_dropped": None,
        "dedup_drop": None,
        "final_count": None,
    }

    # ----------------------------------------------------
    # 2. 文本清洗（HTML 去除 + 繁简转换 + 空白规整）
    # ----------------------------------------------------
    print("信息：开始进行多核并行文本清洗。")
    try:
        df["text"] = df["text"].parallel_apply(clean_text_content)
    except AttributeError:
        print("警告：parallel_apply 不可用，将使用 progress_apply（单进程）替代。")
        tqdm.pandas()
        df["text"] = df["text"].progress_apply(clean_text_content)

    # 去除清洗后为空的文本，并按最小长度阈值过滤
    df.dropna(subset=["text"], inplace=True)
    df["text_len"] = df["text"].str.len()

    before_len_filter = len(df)
    df = df[df["text_len"] >= MIN_LEN]
    after_len_filter = len(df)

    print(f"信息：文本清洗和长度过滤后样本数: {after_len_filter:,}")
    print(
        f"信息：因长度不足（len < {MIN_LEN}）或清洗为空被过滤的样本数: "
        f"{before_len_filter - after_len_filter:,}"
    )

    lineage["after_clean_and_minlen"] = int(after_len_filter)
    lineage["length_filter_drop"] = int(before_len_filter - after_len_filter)

    # ----------------------------------------------------
    # 3. 评分字段标准化与情感标签生成
    # ----------------------------------------------------
    print("\n信息：评分字段统计（清洗后，含非数值内容）:")
    print(df["rating"].value_counts(dropna=False).head(10))

    # 检查 rating 列是否已经是数值类型（即文本标签已被转换）
    sample_rating_values = df["rating"].head(10)
    if sample_rating_values.dtype in ['int64', 'int32', 'float64', 'float32']:
        # 如果已经是数值类型，直接使用
        print("\n信息：rating 列已是数值类型，跳过数值化转换。")
        # 确保 rating 值在 [0, 2] 范围内（对应 消极, 中性, 积极）
        df = df[(df["rating"] >= 0) & (df["rating"] <= 2)]
        # 直接将 rating 作为 label
        df["label"] = df["rating"].astype("int8")
    else:
        # 如果仍是非数值类型，进行数值化转换
        # 将 rating 转换为浮点数，不可转换的内容视为缺失并剔除
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])

        # 显式约束评分范围：豆瓣评分通常为 1–5 分
        df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

        print("\n信息：数值化并约束到 [1, 5] 区间后的评分分布:")
        print(df["rating"].value_counts().sort_index())

        # 将评分映射为情感三分类标签：
        #   rating < 3 → label = 0 （负向）
        #   rating == 3 → label = 1 （中性）
        #   rating > 3 → label = 2 （正向）
        df["label"] = df["rating"].apply(lambda x: 0 if x < 3 else (1 if x == 3 else 2)).astype("int8")
    
    lineage["after_rating_valid"] = int(len(df))

    # 记录中性样本数量
    neutral_count = (df["label"] == 1).sum()
    lineage["neutral_3_count"] = int(neutral_count)

    print("\n信息：情感标签分布（0 = 消极/负向，1 = 中性，2 = 积极/正向）:")
    print(df["label"].value_counts().to_dict())

    # ----------------------------------------------------
    # 4. 文本去重（防止信息泄露）
    # ----------------------------------------------------
    # 若相同新闻文本出现在多个样本中，可能在划分 Train/Val/Test 时产生文本泄露。
    # 因此，在构建训练集之前，在全局范围内按文本内容进行去重。
    before_dedup = len(df)
    df.drop_duplicates(subset=["text"], inplace=True)
    after_dedup = len(df)

    print("\n信息：文本内容去重完成。")
    print(f"    去重前样本数: {before_dedup:,}")
    print(f"    去重后样本数: {after_dedup:,}")
    print(f"    因文本完全相同而删除的重复样本数: {before_dedup - after_dedup:,}")

    lineage["dedup_drop"] = int(before_dedup - after_dedup)

    # ----------------------------------------------------
    # 5. 类别平衡处理（可选，下采样）
    # ----------------------------------------------------
    balance_info = {
        "enable_balance": ENABLE_BALANCE,
        "before_balance": df["label"].value_counts().to_dict(),
        "after_balance": None,
        "balance_method": None,
    }

    if ENABLE_BALANCE:
        print("\n信息：正在检查并处理类别不平衡。")
        counts = df["label"].value_counts()
        print(f"    原始标签分布: {counts.to_dict()}")

        min_count = counts.min()
        max_count = counts.max()

        # 若最大类与最小类样本数比例超过 1.5，则认为存在明显不平衡，执行下采样
        if max_count / min_count > 1.5:
            print(
                f"信息：检测到显著类别不平衡，将通过下采样将各类样本数统一为 {min_count} 条。"
            )
            df_min = df[df["label"] == 0].sample(n=int(min_count), random_state=SEED)
            df_maj = df[df["label"] == 1].sample(n=int(min_count), random_state=SEED)
            df = (
                pd.concat([df_min, df_maj])
                .sample(frac=1, random_state=SEED)
                .reset_index(drop=True)
            )

            print(f"信息：下采样后总样本数: {len(df):,}")
            print(f"    下采样后标签分布: {df['label'].value_counts().to_dict()}")

            balance_info["after_balance"] = df["label"].value_counts().to_dict()
            balance_info["balance_method"] = f"downsample_to_{int(min_count)}"
        else:
            print("信息：样本分布已较为平衡，无需执行下采样处理。")
    else:
        print(
            "\n信息：已关闭类别平衡处理（ENABLE_BALANCE = False），保留真实标签分布。"
        )

    # ----------------------------------------------------
    # 6. 文本长度统计与可视化（用于确定 MAX_LEN）
    # ----------------------------------------------------
    print("\n信息：正在计算文本长度分布，用于确定模型的最大序列长度 MAX_LEN。")

    doc_lengths = df["text"].str.len()

    # 计算若干关键分位数和统计指标
    mean_len = float(doc_lengths.mean())
    p50 = int(np.percentile(doc_lengths, 50))
    p90 = int(np.percentile(doc_lengths, 90))
    p95 = int(np.percentile(doc_lengths, 95))
    p99 = int(np.percentile(doc_lengths, 99))
    max_len_stats = int(doc_lengths.max())
    min_len_stats = int(doc_lengths.min())

    print(f"    平均长度 (mean): {mean_len:.2f}")
    print(f"    中位数 (P50): {p50}")
    print(f"    P90: {p90}")
    print(f"    P95 (覆盖 95% 样本): {p95}  ← 推荐作为模型 MAX_LEN")
    print(f"    P99: {p99}")
    print(f"    最小长度: {min_len_stats}")
    print(f"    最大长度: {max_len_stats}")

    # 绘制长度直方图及核密度曲线
    plt.figure(figsize=(10, 6))

    # 为兼容中文显示，设置字体（在 Linux 环境下优先使用文泉驿或黑体）
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    sns.histplot(doc_lengths, bins=100, kde=True, color="#4c72b0")
    plt.axvline(
        p95, color="#c44e52", linestyle="--", linewidth=2, label=f"P95 = {p95}"
    )
    plt.title(
        "News Text Length Distribution (Character Count)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Text Length (characters)", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    # x 轴范围限制到 P99 的 1.5 倍以内，以减弱长尾对视觉的干扰
    plt.xlim(0, min(int(p99 * 1.5), max_len_stats))
    plt.legend()
    plt.tight_layout()

    # 创建必要的目录
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    print(f"信息：文本长度分布图已保存至: {PLOT_FILE}")

    # ----------------------------------------------------
    # 7. Train / Val / Test 划分（分层抽样）
    # ----------------------------------------------------
    print(
        "\n信息：正在进行全局随机打乱与分层切分（Train/Val/Test = 70% / 10% / 20%）。"
    )

    gc.collect()

    # 全局打乱，保证后续切分时样本顺序随机化
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 划分前只保留模型训练所需字段
    df = df[["text", "label"]]

    total_final = len(df)
    lineage["final_count"] = int(total_final)

    # 先划分出 80% 的 train_val 和 20% 的 test
    train_val, test = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"],
    )

    # 再在 train_val 内部划分出 70% train 与 10% val（总体 0.8 * 0.125 = 0.1）
    train, val = train_test_split(
        train_val,
        test_size=0.125,
        random_state=SEED,
        stratify=train_val["label"],
    )

    def label_stats(df_part: pd.DataFrame):
        """
        辅助函数：计算某一数据子集的样本规模与标签分布情况。
        返回值会被写入 meta.json，方便在论文中直接引用。
        """
        total = len(df_part)
        counts = df_part["label"].value_counts().to_dict()
        ratios = {int(k): float(v / total) for k, v in counts.items()}
        return {
            "total": int(total),
            "label_counts": {int(k): int(v) for k, v in counts.items()},
            "label_ratio": ratios,
        }

    # ----------------------------------------------------
    # 8. 写入 meta.json（元信息与数据谱系）
    # ----------------------------------------------------
    meta = {
        "data_dir": DATA_DIR,
        "raw_dir": RAW_DIR,
        "min_len_filter": int(MIN_LEN),
        "seed": int(SEED),
        "enable_balance": ENABLE_BALANCE,
        "balance_info": balance_info,
        "length_stats": {
            "mean": mean_len,
            "p50": p50,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "min": min_len_stats,
            "max": max_len_stats,
        },
        "label_distribution_after_clean": {
            int(k): int(v) for k, v in df["label"].value_counts().to_dict().items()
        },
        "splits": {
            "train": label_stats(train),
            "val": label_stats(val),
            "test": label_stats(test),
        },
        "data_lineage": lineage,
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"信息：元信息与数据谱系已保存至: {META_FILE}")

    # ----------------------------------------------------
    # 9. 保存最终数据集 CSV 文件
    # ----------------------------------------------------
    print("\n信息：正在保存切分后的 Train/Val/Test 数据集。")
    train.to_csv(TRAIN_FILE, index=False)
    val.to_csv(VAL_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    print("\n信息：数据预处理流程全部完成。")
    print(f"    Train 样本数: {len(train):,}")
    print(f"    Val   样本数: {len(val):,}")
    print(f"    Test  样本数: {len(test):,}")
    print(f"建议：在后续模型代码中设置 MAX_LEN = P95 = {p95}。")


if __name__ == "__main__":
    process_data()