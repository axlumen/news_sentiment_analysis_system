"""
main.py

中文新闻情感分析项目 - 统一命令行入口。

本文件作为项目的主程序入口，通过子命令方式调用各个功能模块。
所有核心逻辑均位于 model/ 目录下，本文件仅负责命令行参数解析与模块调度。

使用方式：
    python main.py init_data       # 数据预处理（清洗、划分 train/val/test）
    python main.py preprocess      # 构建词表（基于训练集）
    python main.py train           # 训练 Bi-LSTM + Attention 模型
    python main.py predict         # 测试 Bi-LSTM 模型并生成可视化
    python main.py plot_curve      # 绘制训练曲线图

项目结构：
    model/
    ├── main.py                 # 本文件，统一入口
    ├── bilstm_attention.py     # Bi-LSTM + Attention 模型定义
    ├── train.py                # 模型训练逻辑
    ├── predict.py              # 模型测试与可视化
    ├── data/                   # 数据处理模块
    │   ├── __init__.py
    │   ├── init_data.py        # 数据预处理与划分
    │   └── preprocess.py       # 词表构建与文本预处理
    └── utils/                  # 工具模块
        ├── __init__.py
        ├── dataset.py          # 数据集封装
        └── plot_curve.py       # 训练曲线绘制
"""

import sys
import os




def run_init_data() -> None:
    """
    执行数据预处理流程。

    调用 data/init_data.py 中的 process_data() 函数，
    完成原始新闻数据清洗、文本过滤、情感标签生成、数据划分等操作。
    """
    print("=" * 60)
    print("执行命令：init_data - 数据预处理")
    print("=" * 60)
    import sys
    import os
    # 添加当前目录到 Python 路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.data.init_data import process_data
    process_data()


def run_preprocess() -> None:
    """
    执行词表构建流程。

    调用 data/preprocess.py 的脚本入口逻辑，
    基于训练集进行分词与词频统计，生成 vocab.pkl 词表文件。
    """
    print("=" * 60)
    print("执行命令：preprocess - 构建词表")
    print("=" * 60)

    # 直接导入并执行 preprocess.py 的主逻辑
    import sys
    import os
    import json
    import pandas as pd
    # 添加当前目录到 Python 路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.data.preprocess import TextPreprocessor

    # 计算项目根目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 路径配置（位于 data/processed/）
    TRAIN_CSV = os.path.join(BASE_DIR, "data", "processed", "train.csv")
    META_JSON = os.path.join(BASE_DIR, "data", "processed", "meta.json")
    VOCAB_PKL = os.path.join(BASE_DIR, "data", "processed", "vocab.pkl")

    # 检查训练集是否存在
    if not os.path.exists(TRAIN_CSV):
        print(f"错误：找不到训练集文件 {TRAIN_CSV}。")
        print("提示：请先运行 'python main.py init_data' 生成数据集。")
        return

    # 读取 meta.json 获取 max_len
    if not os.path.exists(META_JSON):
        print(f"警告：未找到元信息文件 {META_JSON}，将使用默认 max_len=128。")
        max_len = 128
    else:
        print(f"信息：读取元信息文件: {META_JSON}")
        with open(META_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        p95 = meta.get("length_stats", {}).get("p95", 128)
        max_len = int(p95)

    print(f"信息：将采用 max_len={max_len} 构建词表。")

    # 读取训练集文本
    print(f"信息：正在读取训练集: {TRAIN_CSV}")
    df_train = pd.read_csv(TRAIN_CSV)

    if "text" not in df_train.columns:
        raise KeyError("训练集中未找到 'text' 列，请确认数据格式。")

    texts = df_train["text"].astype(str).tolist()

    # 构建词表
    processor = TextPreprocessor(max_len=max_len)
    processor.build_vocab(texts, save_path=VOCAB_PKL)

    print("信息：词表构建流程执行完毕。")


def run_train() -> None:
    """
    执行 Bi-LSTM + Attention 模型训练。

    调用 train.py 中的 main() 函数，
    完成模型初始化、训练循环、早停策略、模型保存等操作。
    """
    print("=" * 60)
    print("执行命令：train - 训练 Bi-LSTM + Attention 模型")
    print("=" * 60)
    import sys
    import os
    # 添加当前目录到 Python 路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.train import main
    main()


def run_predict() -> None:
    """
    执行 Bi-LSTM 模型测试与可视化。

    调用 predict.py 的主逻辑，
    在测试集上评估模型并生成分类报告、混淆矩阵、注意力热力图等。
    """
    print("=" * 60)
    print("执行命令：predict - 测试 Bi-LSTM 模型")
    print("=" * 60)
    import sys
    import os
    # 添加当前目录到 Python 路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.predict import load_model, evaluate_test_set, predict_case_studies

    # 加载模型与词表
    model, processor, id2token = load_model()

    # 测试集评估与混淆矩阵
    evaluate_test_set(model)

    # Case Study 可视化 - 新闻情感分析示例
    paper_cases = [
        "这篇报道内容虚假，完全是误导公众的假新闻。",  # 负向
        "政府出台的新政策将极大促进经济发展，受到广泛好评。",  # 正向
        "专家表示，这项研究结果还需要进一步验证。",  # 中性
        "企业季度财报显示，营收同比增长10%，但利润略有下降。",  # 中性
    ]
    predict_case_studies(model, processor, id2token, paper_cases)

    print("\n信息：测试集评估与可视化已全部生成，请查看 data/ 目录。")


def run_plot_curve() -> None:
    """
    绘制训练曲线图。

    调用 utils/plot_curve.py 中的 plot_from_csv() 函数，
    基于训练日志绘制 Loss 和 Accuracy 随 Epoch 变化的曲线。
    """
    print("=" * 60)
    print("执行命令：plot_curve - 绘制训练曲线")
    print("=" * 60)
    import sys
    import os
    # 添加当前目录到 Python 路径
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model.utils.plot_curve import plot_from_csv
    plot_from_csv()


# ============================
# 命令映射表
# ============================
# 将命令字符串映射到对应的执行函数
COMMAND_MAP = {
    "init_data": run_init_data,
    "preprocess": run_preprocess,
    "train": run_train,
    "predict": run_predict,
    "plot_curve": run_plot_curve,
}


def main() -> None:
    """
    主入口函数。

    解析命令行参数并调用对应的子命令处理函数。
    当没有参数时，默认执行完整流程。
    """
    # 检查命令行参数
    if len(sys.argv) < 2:
        # 当没有参数时，执行完整流程
        print("=" * 60)
        print("执行完整流程：数据预处理 → 构建词表 → 训练模型 → 测试模型 → 绘制曲线")
        print("=" * 60)
        
        print("\n[Step 1] 数据预处理")
        run_init_data()
        
        print("\n[Step 2] 构建词表")
        run_preprocess()
        
        print("\n[Step 3] 训练模型")
        run_train()
        
        print("\n[Step 4] 测试模型")
        run_predict()
        
        print("\n[Step 5] 绘制训练曲线")
        run_plot_curve()
        
        print("\n" + "=" * 60)
        print("完整流程执行完毕！")
        print("=" * 60)
        sys.exit(0)

    command = sys.argv[1].lower()

    # 查找并执行对应的命令
    if command in COMMAND_MAP:
        try:
            COMMAND_MAP[command]()
        except KeyboardInterrupt:
            print("\n\n信息：用户中断执行。")
            sys.exit(130)
        except Exception as e:
            print(f"\n错误：执行命令 '{command}' 时发生异常：{e}")
            raise
    else:
        print(f"错误：未知命令 '{command}'。")
        sys.exit(1)


# ============================
# 脚本入口
# ============================
if __name__ == "__main__":
    main()