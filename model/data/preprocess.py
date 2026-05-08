import os
import json
import pickle
from typing import List, Dict

from collections import Counter

import pandas as pd
import jieba
from tqdm import tqdm
import torch

"""
preprocess.py

本脚本负责基于清洗后的训练集构建中文分词词表，并提供文本序列化工具类，主要包括：
1. 从 init_data.py 生成的 train.csv / meta.json 中读取新闻文本数据与长度统计信息。
2. 使用结巴分词 (Jieba) 在全量训练语料上进行分词与词频统计。
3. 基于词频构建高频词表（限定最大大小 max_vocab_size），并保留 <PAD> / <UNK> 特殊符号。
4. 将词表结构及部分统计信息序列化保存为 vocab.pkl，以便后续模型训练和推理阶段加载复用。
5. 提供 TextPreprocessor 类，用于：
   - 新闻文本分词与 ID 映射（text_to_sequence）
   - 批量文本序列化为张量（process_batch）

设计要点：
- 仅使用训练集构建词表，避免利用验证集/测试集信息导致数据泄露。
- 使用 meta.json 中的 P95 文本长度作为默认 max_len，保证与数据预处理阶段配置一致。
- 在分词统计阶段采用批处理策略，兼顾语料规模和内存占用。
"""



class TextPreprocessor:
    """
    新闻文本预处理核心类。

    功能：
        1. 基于全量训练语料构建词表（分词 + 词频统计）。
        2. 提供单句 / 批量新闻文本到 ID 序列的转换接口。

    词表约定：
        - vocab 是一个从“分词结果（字符串）到整数 ID”的映射字典。
        - 0 号 ID 预留给 <PAD>，用于序列填充。
        - 1 号 ID 预留给 <UNK>，用于表示 OOV（词表外）词汇。
    """


    def __init__(self, max_vocab_size: int = 30000, max_len: int = 128):
        """
        初始化新闻文本预处理器。

        参数：
            max_vocab_size (int):
                词表最大容量（包含 <PAD> 和 <UNK>）。
                对于中文新闻情感分析任务，30,000 左右的词表通常可以覆盖绝大多数常见用词。
            max_len (int):
                文本序列最大长度（单位：分词后的 token 数）。
                推荐根据 init_data.py 中统计得到的 P95 文本长度进行设置。
        """

        self.max_vocab_size = max_vocab_size
        self.max_len = max_len

        # 特殊符号及对应 ID
        # 约定：
        #   - <PAD> 用于对短序列进行填充，以对齐 batch 内长度。
        #   - <UNK> 用于表示未登录词（不在词表中的 token）。
        self.vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

        self.pad_token_id: int = 0
        self.unk_token_id: int = 1

    def build_vocab(self, texts: List[str], save_path: str = None):
        """
        基于给定新闻文本列表构建词表（Vocabulary）。

        构建流程：
            1. 对全量训练语料分批进行分词与词频统计，得到 word_counts。
            2. 根据词频从高到低选取前 (max_vocab_size - 2) 个词，预留 <PAD> 和 <UNK>。
            3. 将选出的词按出现顺序依次分配新的整数 ID，构建 vocab 映射。
            4. 如指定 save_path，则将 vocab 及部分统计信息序列化保存为 pickle 文件。

        优化策略：
            - 采用批量拼接文本的方式，将一批新闻合并为一个大字符串，再调用 jieba.cut 进行分词，
              能够有效降低 Python 调用开销并提升分词效率。
            - 使用 Counter 流式更新词频，适合大规模语料。
        """

        total_count = len(texts)
        print(f"[Step 2] 开始构建词表（训练语料规模: {total_count:,} 条）。")

        # 全局词频统计器
        word_counts = Counter()

        # 批大小（可根据内存情况调整）
        batch_size = 100_000

        # 分词说明：
        #   - 此处不依赖 jieba 的并行接口，以保证在不同平台上的行为一致性。
        #   - 若确需进一步加速，可在后续工作中考虑结合 multiprocessing 自行实现并行。
        for start in tqdm(
            range(0, total_count, batch_size),
            desc="流式分词统计中",
            unit="batch",
        ):
            end = start + batch_size
            batch_texts = texts[start:end]

            # 过滤掉非字符串，并通过换行符拼接为一个大文本块
            large_text = "\n".join(
                [str(t) for t in batch_texts if isinstance(t, str)]
            )

            # 使用 jieba.cut（生成器形式），避免中间结果列表过大占用内存
            tokens = (w for w in jieba.cut(large_text))
            # 过滤掉空白 token
            tokens = [w for w in tokens if w.strip()]

            # 更新全局词频统计
            word_counts.update(tokens)

        print("信息：词频统计完成，正在根据词频截断构建词表。")

        # 只保留出现频率最高的 Top-K 个词
        # 注意：需要为 <PAD> 和 <UNK> 预留两个位置，因此此处减 2
        most_common = word_counts.most_common(self.max_vocab_size - 2)

        # 以当前 vocab 长度为起点为新词分配 ID，确保 <PAD>=0, <UNK>=1 不被覆盖
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)

        print("信息：词表构建完成。")
        print(f"    原始词汇量（unique tokens）: {len(word_counts):,}")
        print(f"    最终词表大小（vocab size）: {len(self.vocab):,}")
        # 如需在论文中报告词汇覆盖率，可根据需要解注释以下一行：
        # print(f"    词汇覆盖率: {(len(self.vocab) / len(word_counts)) * 100:.2f}%")

        if save_path:
            # 为提高后续分析便利性，可以将基础统计信息一并保存
            save_obj = {
                "vocab": self.vocab,
                "max_vocab_size": self.max_vocab_size,
                "max_len": self.max_len,
                # 只保留最常见若干词的词频信息，便于后续可选的可视化/分析
                "top_k_most_common": most_common[:1000],
            }

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(save_obj, f)

            print(f"信息：词表及统计信息已序列化保存至: {save_path}")

    def load_vocab(self, vocab_path: str):
        """
        从指定路径加载已构建的词表。

        兼容两种格式：
            1. 仅保存 vocab 字典的旧格式。
            2. 保存包含 {vocab, max_vocab_size, max_len, top_k_most_common} 等字段的新格式。
        """
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"找不到词表文件: {vocab_path}，请先运行本脚本以构建词表。"
            )

        with open(vocab_path, "rb") as f:
            obj = pickle.load(f)

        # 新版格式：包含 vocab 字段的 dict
        if isinstance(obj, dict) and "vocab" in obj:
            self.vocab = obj["vocab"]
            # 如有记录 max_len / max_vocab_size，则更新到当前实例
            self.max_vocab_size = int(obj.get("max_vocab_size", self.max_vocab_size))
            self.max_len = int(obj.get("max_len", self.max_len))
        else:
            # 兼容旧版：直接存储 vocab 字典
            self.vocab = obj

        print(f"信息：已加载词表，共 {len(self.vocab)} 个 token（含特殊符号）。")

    def text_to_sequence(self, text: str) -> List[int]:
        """
        将单条新闻文本转换为定长的 ID 序列（文本 → token → ID → padding）。

        处理步骤：
            1. 使用 jieba.lcut 对输入新闻文本进行分词。
            2. 若分词结果长度超过 max_len，则保留前 max_len 个 token（截断）。
            3. 将每个 token 映射为对应的整数 ID，不在词表中的 token 映射为 <UNK>。
            4. 若序列长度不足 max_len，则在末尾补充 <PAD>，直至长度达到 max_len。

        参数：
            text (str): 输入的原始新闻文本字符串。

        返回：
            List[int]: 长度固定为 max_len 的整数 ID 序列。
        """

        # 确保输入为字符串类型
        text = str(text)

        # 单条文本分词不启用并行，以避免不必要的多进程开销
        words = jieba.lcut(text)

        # 截断过长序列
        if len(words) > self.max_len:
            words = words[:self.max_len]

        # 将 token 转换为 ID；OOV 词映射到 <UNK>
        seq = [self.vocab.get(w, self.unk_token_id) for w in words]

        # 对短序列进行 <PAD> 填充
        if len(seq) < self.max_len:
            pad_length = self.max_len - len(seq)
            seq.extend([self.pad_token_id] * pad_length)

        return seq

    def process_batch(self, texts: List[str]) -> torch.Tensor:
        """
        将一批新闻文本转换为形状为 [batch_size, max_len] 的张量。

        参数：
            texts (List[str]): 新闻文本列表。

        返回：
            torch.Tensor: 整型张量，dtype=torch.long，形状为 [batch_size, max_len]。
        """

        sequences = [self.text_to_sequence(t) for t in texts]
        return torch.tensor(sequences, dtype=torch.long)


# ============================
# 脚本入口：仅构建词表
# ============================
if __name__ == "__main__":
    """
    当以脚本方式直接运行本文件时，将执行以下流程：
        1. 从项目根目录下的 data/train.csv 读取训练集新闻文本。
        2. 从 data/meta.json 中读取文本长度统计信息，获取 P95 作为 max_len。
        3. 基于训练集的 text 列构建词表，并将结果保存为 data/vocab.pkl。
    """

    # 计算项目根目录（假设本文件位于 model/data/ 目录下）
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 训练集与元信息文件路径（位于 data/processed/）
    TRAIN_CSV = os.path.join(BASE_DIR, "data", "../dataset/processed", "train.csv")
    META_JSON = os.path.join(BASE_DIR, "data", "../dataset/processed", "meta.json")
    VOCAB_PKL = os.path.join(BASE_DIR, "data", "../dataset/processed", "vocab.pkl")

    # 基础存在性检查
    if not os.path.exists(TRAIN_CSV):
        print(f"错误：找不到训练集文件 {TRAIN_CSV}。")
        print("提示：请先运行 init_data.py 生成 train.csv / val.csv / test.csv。")
        raise SystemExit(1)

    if not os.path.exists(META_JSON):
        print(f"警告：未找到元信息文件 {META_JSON}，将使用默认 max_len=128。")
        meta = {}
    else:
        print(f"信息：读取元信息文件: {META_JSON}")
        with open(META_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # 从 meta.json 中读取长度统计信息，优先使用 P95 作为 max_len
    default_max_len = 128
    p95 = meta.get("length_stats", {}).get("p95", default_max_len)
    try:
        max_len = int(p95)
    except (TypeError, ValueError):
        max_len = default_max_len

    print(f"信息：从 meta.json 中获取的 P95={p95}，将采用 max_len={max_len} 构建词表。")

    # 读取训练集文本
    print(f"信息：正在读取训练集: {TRAIN_CSV}")
    df_train = pd.read_csv(TRAIN_CSV)

    if "text" not in df_train.columns:
        raise KeyError("训练集中未找到 'text' 列，请确认 init_data.py 的输出格式是否正确。")

    texts = df_train["text"].astype(str).tolist()

    # 实例化文本预处理器并构建词表
    processor = TextPreprocessor(max_len=max_len)
    processor.build_vocab(texts, save_path=VOCAB_PKL)

    print("信息：词表构建流程执行完毕。后续模型训练脚本可加载 vocab.pkl 并复用 TextPreprocessor。")