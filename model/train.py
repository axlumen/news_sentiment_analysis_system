"""
src/train.py

Bi-LSTM + Attention 情感分析模型的训练模块。

本模块负责：
    1. 定义训练配置类 Config，集中管理数据路径、模型超参数、训练参数等。
    2. 提供单轮训练函数 train_one_epoch 和验证评估函数 evaluate。
    3. 实现完整的训练流程 main()，包括：
       - 随机种子设置（保证实验可复现）
       - 词表加载与数据集构建
       - 模型初始化、优化器与学习率调度器配置
       - 训练循环（含早停机制 Early Stopping）
       - 训练日志保存（CSV 格式，便于后续绘图分析）

设计要点：
    - 使用 BCEWithLogitsLoss 搭配手动正样本加权，以缓解类别不平衡问题。
    - 支持自动混合精度训练（AMP），在 GPU 上可显著加速训练过程。
    - 验证集 F1 分数作为早停与模型保存的依据，而非单纯的准确率。
"""

import os
import time
import random
import pickle
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

import sys
import os
# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 尝试导入 torch.amp，如果不可用则使用兼容方案
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    # 对于旧版本 PyTorch，使用空实现
    class GradScaler:
        def __init__(self, enabled=True):
            self.enabled = enabled
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass
    
    from contextlib import nullcontext
    autocast = nullcontext

# 导入模型与数据集相关组件
from bilstm_attention import SentimentModel
from utils.dataset import SentimentDataset, collate_fn, calculate_pos_weight


class Config:
    """
    训练配置类。

    该类集中定义了数据路径、模型超参数、训练参数以及随机种子等配置，
    便于在论文中复述实验设置，并支持后续实验的统一修改。

    属性说明：
        TRAIN_CSV (str): 训练集 CSV 文件路径。
        VAL_CSV (str): 验证集 CSV 文件路径。
        VOCAB_PATH (str): 词表文件路径（由 preprocess.py 生成）。
        LOG_SAVE_PATH (str): 训练日志保存路径。
        MODEL_SAVE_PATH (str): 最佳模型权重保存路径。
        MAX_LEN (int): 输入序列最大长度（基于 P95 统计结果）。
        DEVICE (torch.device): 计算设备（自动检测 GPU 可用性）。
        SEED (int): 全局随机种子。
        EMBEDDING_DIM (int): 词嵌入维度。
        HIDDEN_DIM (int): LSTM 隐藏层维度。
        OUTPUT_DIM (int): 输出维度（二分类为 1）。
        N_LAYERS (int): LSTM 层数。
        DROPOUT (float): Dropout 比例。
        BATCH_SIZE (int): 批次大小。
        LEARNING_RATE (float): 初始学习率。
        EPOCHS (int): 最大训练轮数。
        PATIENCE (int): 早停耐心轮数。
    """

    # ==================== 数据与模型保存路径 ====================
    # 路径调整：数据文件位于 dataset/processed/，日志位于 dataset/reports/，模型位于 checkpoints/
    TRAIN_CSV: str = "dataset/processed/train.csv"
    VAL_CSV: str = "dataset/processed/val.csv"
    VOCAB_PATH: str = "dataset/processed/vocab.pkl"
    LOG_SAVE_PATH: str = "dataset/reports/training_log.csv"
    MODEL_SAVE_PATH: str = "checkpoints/best_model.pth"

    # ==================== 序列长度配置 ====================
    # 基于 init_data.py 中的 P95 统计结果设置
    MAX_LEN: int = 64  # 减小序列长度以减少内存使用

    # ==================== 硬件与随机性设置 ====================
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED: int = 42

    # ==================== DataLoader 相关参数 ====================
    NUM_WORKERS: int = 2  # 减小worker数量以减少内存使用
    PIN_MEMORY: bool = False  # 在CPU上训练时关闭pin_memory

    # ==================== 模型结构参数 ====================
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256
    OUTPUT_DIM: int = 3  # 三分类：0-负向，1-中性，2-正向
    N_LAYERS: int = 2
    DROPOUT: float = 0.5

    # ==================== 训练超参数 ====================
    BATCH_SIZE: int = 64  # 减小批量大小以避免内存不足
    LEARNING_RATE: float = 1e-3
    EPOCHS: int = 20
    PATIENCE: int = 5  # 早停（Early Stopping）耐心轮数


def set_seed(seed: int) -> None:
    """
    设置 Python、NumPy 与 PyTorch 的随机种子，以提升实验的可复现性。

    该函数会同时设置：
        - Python 内置 random 模块的种子
        - NumPy 的随机种子
        - PyTorch CPU 和 GPU 的随机种子
        - cuDNN 的确定性模式（牺牲部分性能以换取完全可复现性）

    参数：
        seed (int): 随机种子值。

    返回：
        None
    """
    # Python 内置随机模块
    random.seed(seed)

    # NumPy 随机种子
    np.random.seed(seed)

    # PyTorch CPU 随机种子
    torch.manual_seed(seed)

    # PyTorch GPU 随机种子（如果使用 CUDA）
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置 cuDNN 为确定性模式，以保证完全可复现
    # 注意：这会禁用 cuDNN 的自动调优，可能略微影响性能
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"信息：随机种子已设置为 {seed}。")


def count_parameters(model: nn.Module) -> int:
    """
    统计模型中可训练参数的总数。

    该函数遍历模型的所有参数，仅统计 requires_grad=True 的参数数量。
    结果可用于论文中报告模型复杂度。

    参数：
        model (nn.Module): 待统计的 PyTorch 模型。

    返回：
        int: 模型中需要梯度更新的参数总数量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: nn.Module,
    iterator: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> Tuple[float, float]:
    """
    执行单个训练轮次（epoch）。

    该函数完成以下操作：
        1. 将模型设为训练模式（启用 Dropout 等）。
        2. 遍历所有 batch，执行前向传播、损失计算、反向传播与参数更新。
        3. 支持自动混合精度（AMP）训练，在 GPU 上可显著加速。

    参数：
        model (nn.Module): 待训练模型。
        iterator (DataLoader): 训练数据的 DataLoader。
        optimizer (Optimizer): 优化器（如 Adam）。
        criterion (Module): 损失函数（此处为 CrossEntropyLoss）。
        device (torch.device): 计算设备（CPU 或 GPU）。
        scaler (GradScaler): AMP 相关的梯度缩放器；在 CPU 上可以设置 enabled=False。
        use_amp (bool): 是否启用自动混合精度（AMP）。

    返回：
        Tuple[float, float]:
            - 平均训练损失（scalar）。
            - 训练集准确率（0~1）。
    """
    # 将模型设为训练模式
    model.train()

    # 初始化统计变量
    epoch_loss = 0.0
    correct = 0
    total = 0

    # 根据是否使用 GPU，决定是否使用 autocast 上下文
    if use_amp:
        amp_context = autocast(device_type="cuda")
    else:
        # 在 CPU 上使用空上下文管理器
        from contextlib import nullcontext
        amp_context = nullcontext()

    # 遍历所有 batch 进行训练
    for texts, labels in tqdm(iterator, desc="Training", leave=False):
        # 将数据移动到指定设备（non_blocking=True 可与 pin_memory 配合加速）
        texts = texts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()  # 三分类标签需要是长整型

        # 梯度清零
        optimizer.zero_grad()

        with amp_context:
            # 前向传播：预测输出为 Logits（未过 Softmax）
            predictions, _ = model(texts)

            # 计算 CrossEntropyLoss
            loss = criterion(predictions, labels)

        # 反向传播与参数更新
        if use_amp:
            # 使用 GradScaler 进行缩放，以适配混合精度训练
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 在 CPU 或未启用 AMP 的情况，使用常规反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 累计损失
        epoch_loss += loss.item()

        # 计算预测类别用于训练准确率
        _, predicted_classes = torch.max(predictions, dim=1)

        correct += (predicted_classes == labels).sum().item()
        total += labels.size(0)

    # 计算平均损失与准确率
    avg_loss = epoch_loss / len(iterator)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    iterator: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    在验证集或测试集上评估模型性能。

    该函数完成以下操作：
        1. 将模型设为评估模式（禁用 Dropout 等）。
        2. 在 torch.no_grad() 上下文中遍历所有 batch，执行前向传播。
        3. 收集所有预测结果与真实标签，计算准确率与 F1 分数。

    参数：
        model (nn.Module): 待评估模型。
        iterator (DataLoader): 验证或测试数据的 DataLoader。
        criterion (Module): 损失函数（CrossEntropyLoss）。
        device (torch.device): 计算设备。

    返回：
        Tuple[float, float, float]:
            - 平均损失（scalar）。
            - 准确率（0~1）。
            - F1 分数（0~1，多分类）。
    """
    # 将模型设为评估模式
    model.eval()

    # 初始化统计变量
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    # 禁用梯度计算以节省内存和加速推理
    with torch.no_grad():
        for texts, labels in iterator:
            # 将数据移动到指定设备
            texts = texts.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()  # 三分类标签需要是长整型

            # 前向传播
            predictions, _ = model(texts)

            # 计算损失
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()

            # 将 Logits 转换为类别预测
            _, preds = torch.max(predictions, dim=1)

            # 收集预测结果与真实标签
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # 计算评估指标
    avg_loss = epoch_loss / len(iterator)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")  # 多分类使用 macro F1

    return avg_loss, accuracy, f1


def main() -> None:
    """
    训练主函数。

    该函数是 Bi-LSTM + Attention 模型训练的完整流程入口，包括：
        1. 设置随机种子与设备。
        2. 加载词表与构建数据集、DataLoader。
        3. 初始化模型、优化器、损失函数与学习率调度器。
        4. 执行训练与验证循环，使用早停策略保存最佳模型。
        5. 将训练过程日志保存为 CSV 文件，便于后续分析与可视化。

    返回：
        None
    """
    # ==================== 1. 随机种子与设备信息 ====================
    set_seed(Config.SEED)
    device = Config.DEVICE
    use_amp = device.type == "cuda"

    print(f"信息：当前计算设备：{device}。")
    print(f"信息：自动混合精度（AMP）状态：{'启用' if use_amp else '未启用'}。")

    # ==================== 2. 加载词表 ====================
    if not os.path.exists(Config.VOCAB_PATH):
        raise FileNotFoundError(f"词表文件未找到：{Config.VOCAB_PATH}")

    with open(Config.VOCAB_PATH, "rb") as f:
        vocab_obj = pickle.load(f)

    # 兼容两种保存格式：直接 vocab 或 包含 vocab 字段的 dict
    if isinstance(vocab_obj, dict) and "vocab" in vocab_obj:
        vocab = vocab_obj["vocab"]
    else:
        vocab = vocab_obj

    vocab_size = len(vocab)
    print(f"信息：词表大小（vocab size）为：{vocab_size}。")

    # ==================== 3. 构建训练集与验证集 DataLoader ====================
    train_data = SentimentDataset(
        csv_path=Config.TRAIN_CSV,
        vocab_path=Config.VOCAB_PATH,
        max_len=Config.MAX_LEN,
    )
    val_data = SentimentDataset(
        csv_path=Config.VAL_CSV,
        vocab_path=Config.VOCAB_PATH,
        max_len=Config.MAX_LEN,
    )

    # 三分类任务不需要正样本权重，使用 CrossEntropyLoss 自动处理类别平衡

    train_loader = DataLoader(
        train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=True,
    )

    # ==================== 4. 初始化模型与优化器 ====================
    model = SentimentModel(
        vocab_size=vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        output_dim=Config.OUTPUT_DIM,
        n_layers=Config.N_LAYERS,
        dropout=Config.DROPOUT,
    ).to(device)

    num_params = count_parameters(model)
    print(f"信息：模型可训练参数总数：{num_params:,}。")

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 使用 CrossEntropyLoss 用于三分类任务
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器：基于验证集损失的自适应调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # AMP 相关的梯度缩放器，在 CPU 上禁用
    scaler = GradScaler(enabled=use_amp)

    # 早停相关变量
    best_metric = 0.0  # 记录最佳验证集 F1
    patience_counter = 0
    history: list[Dict[str, Any]] = []

    print(f"信息：开始训练，共 {Config.EPOCHS} 个 epoch。")

    # ==================== 5. 训练与验证循环 ====================
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # 执行单轮训练
        train_loss, train_acc = train_one_epoch(
            model=model,
            iterator=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        # 在验证集上评估
        val_loss, val_acc, val_f1 = evaluate(
            model=model,
            iterator=val_loader,
            criterion=criterion,
            device=device,
        )

        # 更新学习率调度器
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        # 记录本轮训练日志
        history.append(
            {
                "Epoch": epoch + 1,
                "Train Loss": train_loss,
                "Train Acc": train_acc,
                "Val Loss": val_loss,
                "Val Acc": val_acc,
                "Val F1": val_f1,
                "Time": f"{int(mins)}m {int(secs)}s",
                "LR": current_lr,
            }
        )

        # 打印本轮训练结果
        print(
            f"Epoch: {epoch + 1:02} | Time: {int(mins)}m {int(secs)}s | LR: {current_lr:.6f}"
        )
        print(f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(
            f"\t Val. Loss: {val_loss:.4f} |  Val. Acc: {val_acc * 100:.2f}% | Val F1: {val_f1 * 100:.2f}%"
        )

        # ==================== 早停与模型保存 ====================
        # 使用验证集 F1 作为早停与模型保存的依据
        if val_f1 > best_metric:
            best_metric = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(
                f"\t信息：验证集 F1 提升，已保存当前最优模型（Val F1: {best_metric * 100:.2f}%）。"
            )
        else:
            patience_counter += 1
            print(
                f"\t信息：验证集 F1 未提升（{patience_counter}/{Config.PATIENCE}）。"
            )

        # 检查是否触发早停
        if patience_counter >= Config.PATIENCE:
            print("信息：触发早停机制（Early Stopping），停止训练。")
            break

    # ==================== 6. 保存训练日志 ====================
    os.makedirs(os.path.dirname(Config.LOG_SAVE_PATH), exist_ok=True)
    pd.DataFrame(history).to_csv(Config.LOG_SAVE_PATH, index=False)
    print(f"信息：训练日志已保存至：{Config.LOG_SAVE_PATH}。")


# ==================== 脚本入口 ====================
if __name__ == "__main__":
    main()