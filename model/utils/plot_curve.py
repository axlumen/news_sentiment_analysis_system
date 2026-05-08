"""
src/utils/plot_curve.py

训练曲线绘制模块。

本模块基于训练日志 CSV 文件绘制 Loss 和 Accuracy 随 Epoch 变化的曲线图，
用于论文的实验结果展示。

输出文件：
    - dataset/training_curve.png    训练曲线图（左图: Loss，右图: Accuracy）

主要函数：
    - plot_from_csv(): 读取训练日志并绘制曲线图。

设计特点：
    - 使用 Seaborn whitegrid 风格，适合学术论文插图。
    - 支持中文字体回退策略，兼容不同系统环境。
    - 图像分辨率 300 DPI，适合打印输出。
"""

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns


# ============================
# 路径配置
# ============================
# 日志文件与输出曲线图路径
# 路径调整：日志位于 dataset/reports/，图像位于 dataset/
LOG_PATH = "../dataset/reports/training_log.csv"
SAVE_PATH = "../dataset/training_curve.png"


def plot_from_csv() -> None:
    """
    基于训练日志绘制损失和准确率随 Epoch 变化的曲线图。

    图像设计用于学术论文或技术报告插图：
        - 使用 Seaborn whitegrid 风格，便于打印与阅读。
        - 左图：训练/验证损失曲线（BCE Loss）。
        - 右图：训练/验证准确率曲线（Acc）。
    """
    # 1. 检查日志文件是否存在
    if not os.path.exists(LOG_PATH):
        print(f"错误：找不到训练日志文件 {LOG_PATH}。")
        print("提示：请先运行 main.py 完成模型训练。")
        return

    # 2. 读取训练日志
    print(f"信息：正在读取训练日志：{LOG_PATH}")
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception as e:
        print(f"错误：读取训练日志 CSV 失败：{e}")
        return

    # 必要字段检查
    required_cols = {"Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc"}
    if not required_cols.issubset(df.columns):
        print("错误：训练日志缺少必要列，无法绘制曲线。")
        print(f"期望字段：{required_cols}")
        print(f"实际字段：{set(df.columns)}")
        return

    epochs = df["Epoch"]
    train_loss = df["Train Loss"]
    val_loss = df["Val Loss"]
    train_acc = df["Train Acc"]
    val_acc = df["Val Acc"]

    # 3. 设置绘图风格（Seaborn + 字体配置）
    # 使用 whitegrid 风格，适用于论文中展示
    sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

    # 字体回退策略：优先中文字体，然后退回通用字体
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 4. 创建画布（左右双子图）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左图：Loss 曲线 ---
    ax1.plot(
        epochs,
        train_loss,
        "o-",
        label="Training Loss",
        color="#4c72b0",
        linewidth=2.5,
        markersize=6,
    )
    ax1.plot(
        epochs,
        val_loss,
        "s--",
        label="Validation Loss",
        color="#dd8452",
        linewidth=2.5,
        markersize=6,
    )

    ax1.set_title("Loss Curve", fontsize=16, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=13)
    ax1.set_ylabel("Loss (BCE)", fontsize=13)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fancybox=True)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- 右图：Accuracy 曲线 ---
    ax2.plot(
        epochs,
        train_acc,
        "o-",
        label="Training Acc",
        color="#55a868",
        linewidth=2.5,
        markersize=6,
    )
    ax2.plot(
        epochs,
        val_acc,
        "s--",
        label="Validation Acc",
        color="#c44e52",
        linewidth=2.5,
        markersize=6,
    )

    ax2.set_title("Accuracy Curve", fontsize=16, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=13)
    ax2.set_ylabel("Accuracy", fontsize=13)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, fancybox=True)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 5. 布局与保存
    plt.tight_layout()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")

    print(f"信息：训练曲线图已生成并保存至：{SAVE_PATH}。")
    print("提示：图像已按论文插图需求进行排版，可直接用于 LaTeX 或其他排版系统。")


if __name__ == "__main__":
    plot_from_csv()