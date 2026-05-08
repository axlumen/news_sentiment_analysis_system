"""
predict.py

Bi-LSTM + Attention 新闻情感分析模型的测试与可视化模块。

本脚本用于在测试集上对已训练好的 Bi-LSTM+Attention 模型进行最终评估，并生成：
1. 测试集分类报告（precision / recall / F1 / accuracy），保存为 dataset/test_report.txt。
2. 测试集混淆矩阵图，保存为 dataset/confusion_matrix.png。
3. 若干典型新闻样例（Case Study）的注意力热力图，展示模型在中文新闻句子上的关注位置，
   保存为 dataset/case_study_*.png。

该脚本的设计目标是为论文的"实验结果"与"可解释性分析"部分提供直接可用的图表与统计结果。

主要功能：
    - evaluate_test_set(): 在测试集上批量评估模型，输出分类报告与混淆矩阵。
    - predict_case_studies(): 对给定新闻文本生成注意力热力图，用于可解释性分析。
    - load_model(): 加载已训练的模型权重与词表。
"""

import sys
import os
# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 导入模型与数据集相关组件
from model.bilstm_attention import SentimentModel
from model.utils.dataset import SentimentDataset, collate_fn
from model.data.preprocess import TextPreprocessor


# ============================
# 全局配置
# ============================

CONFIG = {
    # 路径调整：数据位于 dataset/processed/，模型位于 checkpoints/
    "vocab_path": "dataset/processed/vocab.pkl",
    "model_path": "checkpoints/best_model.pth",   # 与 train.py 中的 MODEL_SAVE_PATH 保持一致
    "test_path": "dataset/processed/test.csv",

    # 模型架构参数（需与训练阶段完全一致）
    "embedding_dim": 128,
    "hidden_dim": 256,
    "output_dim": 3,  # 三分类：0-负向，1-中性，2-正向
    "n_layers": 2,
    "dropout": 0.5,

    # 推理相关参数
    "batch_size": 512,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 8,
    "pin_memory": True,
}

# ============================
# 字体与绘图主题设置
# ============================

# 字体候选列表：优先使用 Noto Sans CJK，其次文泉驿/SimHei，最后退回 DejaVu Sans
FONT_CANDIDATES = [
    "Noto Sans CJK SC",   # Arch 上 noto-fonts-cjk 对应字体名
    "WenQuanYi Micro Hei",
    "SimHei",
    "DejaVu Sans",
    "Microsoft YaHei",  # 添加微软雅黑，Windows系统常用字体
    "Arial Unicode MS",  # 添加Arial Unicode MS，支持中文
]

plt.rcParams["font.sans-serif"] = FONT_CANDIDATES
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="white", palette="muted")

# 在 matplotlib 的字体库中查找可用的中文字体，用于后续显式指定给 tick label
_available_fonts = {f.name: f for f in fm.fontManager.ttflist}
CHINESE_FONT = None
for name in FONT_CANDIDATES:
    if name in _available_fonts:
        CHINESE_FONT = fm.FontProperties(fname=_available_fonts[name].fname)
        break
# 若未找到合适中文字体，则 CHINESE_FONT 为 None，此时退回默认字体

# 确保在保存图表时使用支持中文的字体
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = FONT_CANDIDATES


# ============================
# 工具函数
# ============================

def load_vocab_and_processor(vocab_path: str) -> tuple[dict, TextPreprocessor]:
    """
    加载词表文件，并返回 vocab 字典和对应的 TextPreprocessor 实例。

    兼容 preprocess.py 中保存的两种格式：
        1. 仅包含 vocab 的旧格式；
        2. 包含 {vocab, max_vocab_size, max_len, ...} 的新格式。
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词表文件未找到：{vocab_path}")

    with open(vocab_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "vocab" in obj:
        vocab = obj["vocab"]
        max_len = int(obj.get("max_len", 128))
        processor = TextPreprocessor(max_len=max_len)
        processor.vocab = vocab
    else:
        vocab = obj
        processor = TextPreprocessor()  # 使用默认 max_len
        processor.vocab = vocab

    return vocab, processor


def load_model() -> tuple[SentimentModel, TextPreprocessor, dict]:
    """
    加载训练好的新闻情感分析模型与词表，并构建 index → token 的反向映射表。

    返回：
        model: 已加载参数并切换到 eval 模式的 SentimentModel 实例。
        processor: 已加载词表的 TextPreprocessor。
        id2token: 从 token ID 到原始分词字符串的映射字典。
    """

    if not os.path.exists(CONFIG["model_path"]):
        raise FileNotFoundError(
            f"模型文件未找到：{CONFIG['model_path']}，请先运行 main.py 完成训练。"
        )

    print(f"信息：正在加载模型至设备：{CONFIG['device']}。")

    # 1. 加载词表与预处理器
    vocab, processor = load_vocab_and_processor(CONFIG["vocab_path"])
    vocab_size = len(vocab)

    # 2. 构建反向词表（index → token）
    id2token = {idx: token for token, idx in vocab.items()}

    # 3. 初始化模型结构（需与训练阶段完全一致）
    model = SentimentModel(
        vocab_size=vocab_size,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        output_dim=CONFIG["output_dim"],
        n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"],
    )

    # 4. 加载权重参数
    state_dict = torch.load(CONFIG["model_path"], map_location=CONFIG["device"])
    model.load_state_dict(state_dict)
    model.to(CONFIG["device"])
    model.eval()

    print("信息：模型与词表加载完成。")

    return model, processor, id2token


# ============================
# 测试集评估与混淆矩阵
# ============================

def evaluate_test_set(model: SentimentModel) -> None:
    """
    在测试集上进行批量评估，生成分类报告与混淆矩阵图。

    输出：
        - dataset/test_report.txt
        - dataset/confusion_matrix.png
    """

    if not os.path.exists(CONFIG["test_path"]):
        print(f"警告：未找到测试集 {CONFIG['test_path']}，跳过批量评估。")
        return

    print("信息：开始在测试集上进行批量评估。")

    test_data = SentimentDataset(
        csv_path=CONFIG["test_path"],
        vocab_path=CONFIG["vocab_path"],
        max_len=128,  # 与训练阶段保持一致
    )
    test_loader = DataLoader(
        test_data,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"],
    )

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for texts, labels in tqdm(test_loader, desc="评估进度", unit="batch"):
            texts = texts.to(CONFIG["device"], non_blocking=True)
            labels = labels.to(CONFIG["device"], non_blocking=True)

            logits, _ = model(texts)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).long()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.long().cpu().numpy().tolist())

    print("信息：测试集评估完成，正在生成分类报告与混淆矩阵。")

    # 分类报告（precision / recall / F1 / accuracy）
    target_names = ["Negative (负向)", "Neutral (中性)", "Positive (正向)"]
    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("测试集最终评估报告（Final Test Report）")
    print("=" * 60)
    print(report)

    # 保存报告到文件（位于 dataset/reports/）
    report_path = "dataset/reports/test_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"信息：测试集分类报告已保存至：{report_path}。")

    # 混淆矩阵绘制
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        xticklabels=["Pred Neg", "Pred Neu", "Pred Pos"],
        yticklabels=["True Neg", "True Neu", "True Pos"],
        cbar=False,
    )

    ax.set_title("Confusion Matrix on Test Set", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)

    # 统一 tick 字体大小
    ax.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()

    # 路径调整：图像位于 dataset/figures/
    cm_path = "dataset/figures/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"信息：混淆矩阵图已保存至：{cm_path}。")


# ============================
# Case Study 可视化（Attention 热力图）
# ============================

def predict_case_studies(
    model: SentimentModel,
    processor: TextPreprocessor,
    id2token: dict[int, str],
    cases: list[str],
) -> None:
    """
    对给定的若干新闻示例句子生成注意力热力图，用于论文中的 Case Study 展示。

    每个案例将生成一张横向的热力图，横轴为分词后的 token，颜色深浅表示注意力权重大小。
    """

    if not cases:
        return

    print("\n" + "=" * 60)
    print("信息：开始生成 Case Study 可视化图表。")
    print("=" * 60)

    model.eval()

    for i, text in enumerate(cases):
        if not text:
            continue

        print(f"信息：处理案例 Case {i + 1}：{text}")

        # 1. 文本序列化
        seq = processor.text_to_sequence(text)
        tensor = torch.tensor([seq], dtype=torch.long, device=CONFIG["device"])

        # 2. 前向推理，获取 logits 与注意力权重
        with torch.no_grad():
            logits, attn_weights = model(tensor)

        prob = torch.softmax(logits, dim=1).squeeze(0)
        pred_label = torch.argmax(prob).item()
        confidence = prob[pred_label].item()

        # 映射标签到文本
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        label = label_map[pred_label]

        # 3. 解析注意力权重
        weights = attn_weights.squeeze(0).cpu().numpy()  # [seq_len]
        tokens: list[str] = []
        valid_weights: list[float] = []

        for idx, w in zip(seq, weights):
            if idx != 0:  # 忽略 PAD 位置
                token = id2token.get(idx, "<UNK>")
                tokens.append(token)
                valid_weights.append(float(w))

        # 4. 绘制热力图
        if tokens and valid_weights:
            width = max(8.0, len(tokens) * 0.5)
            plt.figure(figsize=(width, 2.8))

            df_attn = pd.DataFrame([valid_weights], columns=tokens)
            ax = sns.heatmap(
                df_attn,
                cmap="Reds",
                annot=True,
                cbar=False,
                fmt=".2f",
                square=True,
                linewidths=0.5,
            )

            # 标题：案例编号 + 预测标签 + 置信度
            ax.set_title(
                f"Case {i + 1}: {label} ({confidence * 100:.1f}%)",
                fontsize=12,
                pad=8,
            )
            ax.set_yticklabels([])
            ax.tick_params(axis="x", labelsize=10)

            # 显式为 x 轴 tick 标签设置中文字体，以避免中文缺字
            if CHINESE_FONT is not None:
                for tick_label in ax.get_xticklabels():
                    tick_label.set_fontproperties(CHINESE_FONT)

            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # 路径调整：图像位于 dataset/figures/
            save_path = f"dataset/figures/case_study_{i + 1}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"信息：Case {i + 1} 的注意力热力图已保存至：{save_path}。")
        else:
            print("信息：该句子在预处理后无有效 token，已跳过。")


# ============================
# 脚本入口
# ============================

if __name__ == "__main__":
    # 1. 加载模型与词表
    model, processor, id2token = load_model()

    # 2. 测试集评估与混淆矩阵
    evaluate_test_set(model)

    # 3. 论文中用于展示的 Case Study 示例
    paper_cases = [
        "这篇报道内容虚假，完全是误导公众的假新闻。",  # 负向
        "政府出台的新政策将极大促进经济发展，受到广泛好评。",  # 正向
        "专家表示，这项研究结果还需要进一步验证。",  # 中性
        "企业季度财报显示，营收同比增长10%，但利润略有下降。",  # 中性
    ]

    predict_case_studies(model, processor, id2token, paper_cases)

    print("\n信息：测试集评估与可视化已全部生成，请查看 dataset/ 目录。")