"""
bilstm_attention.py

Bi-LSTM + Attention 新闻情感分析模型定义。

本模块定义了基于双向 LSTM 和注意力机制的中文新闻情感分析模型（SentimentModel），
是本项目的核心模型实现。

模型架构：
    输入序列（token ID）
        → 词嵌入层（Embedding）
        → 双向 LSTM（Bi-LSTM）
        → 注意力加权池化（Attention Pooling）
        → 全连接层（Linear）
        → 输出 Logits

设计特点：
    1. 双向 LSTM 同时从前向后和从后向前编码序列，
       适合处理中文长句和"虽然...但是..."等转折结构。
    2. 注意力机制对 LSTM 输出进行加权池化，
       使模型能够关注对情感判断最重要的词汇。
    3. 输出为未经 Softmax 的 Logits，
       建议配合 CrossEntropyLoss 使用以获得数值稳定性。

主要类：
    - SentimentModel: Bi-LSTM + Attention 新闻情感分析模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SentimentModel(nn.Module):
    """
    基于 Bi-LSTM + Attention 的中文新闻情感分析模型。

    模型结构：
        输入序列（token ID） → 词嵌入层（Embedding） → 双向 LSTM（Bi-LSTM）
        → 注意力加权池化（Attention Pooling） → 全连接层（Linear） → 输出 Logits。

    设计要点：
        1. 双向 LSTM（Bi-LSTM）
           - 同时从前向后和从后向前编码序列，适合处理中文长句和“虽然...但是...”等转折结构，
             有助于捕获上下文依赖和整体语义。

        2. 改进的注意力机制（Attention）
           - 对于每个时间步 t 的双向隐状态向量 h_t ∈ R^{2H}，计算未激活的注意力打分：
                 e_t = w^T h_t
             与常见形式 e_t = v^T tanh(W h_t) 不同，本实现移除了 tanh 激活，以增强打分的线性可解释性。
           - 在 softmax 之前，对 padding 位置施加掩蔽（Mask），将对应 e_t 置为较大的负值（例如 -1e4），
             使得这些位置在 softmax 后的注意力权重近似为 0：
                 e_t' = e_t                   （若该位置为真实 token）
                 e_t' = -1e4                 （若该位置为 padding）
                 α_t = softmax(e_t')         （沿时间维度归一化）
           - 最终上下文向量 c 通过对所有时间步加权求和得到：
                 c = Σ_t α_t h_t

        3. 输出层与损失函数
           - 本模型的 forward 函数返回未经过 Softmax 的 Logits，
             以便在训练过程中与 nn.CrossEntropyLoss 搭配使用，从而获得数值更稳定的训练过程。
    """


    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        dropout: float,
    ):
        """
        初始化情感分析模型。

        参数：
            vocab_size (int): 词表大小，即可用 token 的种类数。
            embedding_dim (int): 词嵌入向量的维度。
            hidden_dim (int): LSTM 单向隐状态的维度（双向后总维度为 2 * hidden_dim）。
            output_dim (int): 输出维度。对于二分类任务通常为 1（输出单个 Logit）。
            n_layers (int): LSTM 堆叠层数。
            dropout (float): 在 LSTM 与全连接层前使用的 Dropout 比例。
        """
        super(SentimentModel, self).__init__()

        # 词嵌入层，padding_idx=0 与 TextPreprocessor 及 Dataset 中的约定保持一致
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # 双向 LSTM：batch_first=True 使输入张量形状为 [batch, seq_len, embedding_dim]
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        # 注意力打分层：从 2 * hidden_dim 映射到一个标量打分 e_t
        self.attention_weights = nn.Linear(hidden_dim * 2, 1)

        # 全连接分类层：输入为注意力池化后的上下文向量（维度为 2 * hidden_dim）
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def attention_net(
        self, lstm_output: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意力模块。

        参数：
            lstm_output (Tensor): LSTM 的输出张量，形状为 [batch_size, seq_len, 2 * hidden_dim]。
            mask (Tensor): padding 掩蔽张量，形状为 [batch_size, seq_len]，真实 token 位置为 1，padding 位置为 0。

        返回：
            context (Tensor): 注意力加权后的上下文向量，形状为 [batch_size, 2 * hidden_dim]。
            soft_attn_weights (Tensor): 注意力权重，形状为 [batch_size, seq_len]。
        """
        # 1. 计算原始注意力打分，形状 [batch, seq_len, 1]
        attn_scores = self.attention_weights(lstm_output)

        # 将维度从 [batch, seq_len, 1] 压缩为 [batch, seq_len]
        attn_scores = attn_scores.squeeze(2)

        # 2. 对 padding 位置进行掩蔽处理
        if mask is not None:
            # 使用 -1e4 作为被掩蔽位置的打分值，在 FP16 下仍然数值稳定，
            # softmax 后这些位置的注意力权重近似为 0。
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)

        # 3. 沿序列维度进行 softmax 归一化，得到注意力权重
        soft_attn_weights = F.softmax(attn_scores, dim=1)

        # 4. 根据注意力权重对 LSTM 输出进行加权求和，得到上下文向量
        context = torch.sum(
            lstm_output * soft_attn_weights.unsqueeze(2), dim=1
        )  # [batch, 2 * hidden_dim]

        return context, soft_attn_weights

    def forward(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数。

        参数：
            text (Tensor): 输入的 token ID 序列，形状为 [batch_size, seq_len]。
                           其中 0 号 ID 为 <PAD>，用于序列填充。

        返回：
            logits (Tensor): 未经过 Softmax 的 Logits，形状为 [batch_size, output_dim]。
                             在训练阶段建议搭配 nn.CrossEntropyLoss 使用。
            attention_weights (Tensor): 注意力权重，形状为 [batch_size, seq_len]，
                                        可用于可视化模型关注的新闻文本位置。
        """

        # 构建 mask：非 padding 位置为 1，padding 位置为 0
        mask = (text != 0)  # [batch_size, seq_len]

        # 词嵌入并施加 Dropout
        embedded = self.dropout(self.embedding(text))  # [batch, seq_len, embedding_dim]

        # 经过 Bi-LSTM 编码，lstm_output 形状为 [batch, seq_len, 2 * hidden_dim]
        lstm_output, _ = self.lstm(embedded)

        # 注意力加权池化
        attn_output, attn_weights = self.attention_net(lstm_output, mask)

        # 对注意力输出施加 Dropout，再输入到全连接层
        attn_output = self.dropout(attn_output)
        logits = self.fc(attn_output)  # [batch, output_dim]

        # 返回 Logits 和注意力权重（注意力权重可用于可视化与解释）
        return logits, attn_weights