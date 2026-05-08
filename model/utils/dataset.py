"""
src/utils/dataset.py

情感分析数据集与辅助函数模块。

本模块提供 PyTorch Dataset 封装和数据加载相关的辅助函数，包括：
    - SentimentDataset: 自定义 Dataset 类，负责从 CSV 加载数据并转换为模型输入格式。
    - calculate_pos_weight(): 计算正样本权重，用于缓解类别不平衡问题。
    - collate_fn(): 自定义 collate 函数，用于将样本组装为批次。

设计要点：
    - SentimentDataset 依赖 TextPreprocessor 进行文本到 ID 序列的转换。
    - 支持 train.csv / val.csv / test.csv 三种数据集的加载。
    - collate_fn 确保输出张量形状与模型预期一致。
"""

import os
from typing import Tuple, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from model.data.preprocess import TextPreprocessor


class SentimentDataset(Dataset):
    """
    自定义的情感分析数据集类。

    该类负责：
        1. 从 CSV 文件中读取样本（包含 text 与 label 字段）。
        2. 调用 TextPreprocessor 将原始文本转换为定长的 token ID 序列。
        3. 将序列与标签包装为 PyTorch Tensor，供 DataLoader 迭代使用。

    设计要点：
        - 仅负责数据访问与简单预处理，不参与主模型逻辑。
        - 依赖 TextPreprocessor.load_vocab 加载由 preprocess.py 生成的词表。
        - 文本长度 max_len 与预处理阶段保持一致（推荐使用来自 meta.json 中的 P95）。
    """

    def __init__(self, csv_path: str, vocab_path: str, max_len: int = 128):
        """
        初始化数据集。

        参数：
            csv_path (str): 数据集文件路径（如 train.csv / val.csv / test.csv）。
            vocab_path (str): 词表文件路径（vocab.pkl），由 preprocess.py 生成。
            max_len (int): 文本序列的最大长度。应与预处理和模型中使用的配置一致。
        """
        # 文件存在性检查
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据集未找到：{csv_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"词表文件未找到：{vocab_path}")

        # 读取 CSV 数据
        self.df = pd.read_csv(csv_path)

        # 初始化文本预处理器并加载词表
        self.processor = TextPreprocessor(max_len=max_len)
        self.processor.load_vocab(vocab_path)

    def __len__(self) -> int:
        """
        返回数据集中样本的数量。
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据索引返回单条样本。

        参数：
            idx (int): 样本索引。

        返回：
            Tuple[Tensor, Tensor]:
                - input_ids: 形状为 [max_len] 的 LongTensor，表示 token ID 序列。
                - label: 形状为 [] 的 LongTensor，值为 0、1 或 2。
        """
        row = self.df.iloc[idx]

        # 原始文本与标签
        text = str(row["text"])
        label = int(row["label"])

        # 文本转序列 ID
        seq_list = self.processor.text_to_sequence(text)

        input_ids = torch.tensor(seq_list, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)  # 三分类标签使用长整型

        return input_ids, label_tensor


def calculate_pos_weight(csv_path: str) -> torch.Tensor:
    """
    计算正样本权重（pos_weight）。

    该函数用于在损失函数中引入类别权重，以缓解正负样本不平衡问题。
    对于二分类任务，常用的设置为：

        pos_weight = neg_count / pos_count

    在 nn.BCEWithLogitsLoss 中，pos_weight 会放大正类样本的损失项。

    参数：
        csv_path (str): 包含 label 列的训练集 CSV 路径。

    返回：
        torch.Tensor: 标量张量，表示正样本权重。
                      若文件不存在或正样本计数为 0，则返回 1.0（不加权）。
    """
    if not os.path.exists(csv_path):
        # 若数据集不可用，则返回默认权重 1.0（等价于不加权）
        return torch.tensor(1.0, dtype=torch.float)

    df = pd.read_csv(csv_path)

    pos_count = (df["label"] == 1).sum()
    neg_count = (df["label"] == 0).sum()

    if pos_count == 0:
        # 极端情况下若不存在正样本，则退化为不加权
        return torch.tensor(1.0, dtype=torch.float)

    weight = neg_count / pos_count
    return torch.tensor(weight, dtype=torch.float)


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    自定义的 collate 函数，用于将单条样本组装为批次（Batch）。

    该函数主要完成：
        1. 将若干个 [max_len] 的文本序列堆叠为形状 [batch_size, max_len] 的张量。
        2. 将若干个标量标签堆叠为形状 [batch_size] 的张量，
           以匹配 SentimentModel 的输出形状 [batch_size, 3]。

    参数：
        batch (List[Tuple[Tensor, Tensor]]):
            来自 Dataset.__getitem__ 的列表，每个元素为 (input_ids, label)。

    返回：
        Tuple[Tensor, Tensor]:
            - texts_tensor: LongTensor，形状为 [batch_size, max_len]。
            - labels_tensor: LongTensor，形状为 [batch_size]。
    """
    # 解包 batch 列表
    texts, labels = zip(*batch)

    # 堆叠文本序列 [B, max_len]
    texts_tensor = torch.stack(texts, dim=0)

    # 堆叠标签 [B]
    labels_tensor = torch.stack(labels, dim=0)

    return texts_tensor, labels_tensor