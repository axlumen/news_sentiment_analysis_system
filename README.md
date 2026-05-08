# News Sentiment Analysis System

基于 Django + PyTorch 的中文新闻情感分析平台，实现从新闻数据采集、预处理、情感分析到可视化展示的完整流程。

## ✨ Features

- **新闻数据采集**：多板块新浪新闻自动爬取（综合、财经、科技、体育、娱乐等）
- **数据预处理**：智能文本清洗、去重、标准化处理
- **情感分析**：Bi-LSTM + Attention 三分类模型（正面、中性、负面）
- **统计分析**：多维度情感统计与可视化
- **用户管理**：完整的用户注册、登录、个人中心功能
- **可视化仪表盘**：情感分布饼图、趋势图、词云等

## 🛠 Technology Stack

| 层次 | 技术 | 版本 |
|------|------|------|
| Web框架 | Django | 5.2.12 |
| 数据库 | MySQL | 8.0+ |
| 深度学习 | PyTorch | 2.0+ |
| 分词 | jieba | 0.42.1 |
| 数据处理 | pandas | 2.0+ |
| 可视化 | Chart.js | 5.0+ |
| 爬虫 | requests + parsel | - |

## 📁 Project Structure

```
news_sentiment_analysis/
├── news_sentiment_analysis/      # Django项目配置
│   ├── settings.py              # 全局配置
│   ├── urls.py                  # 路由配置
│   └── wsgi.py                  # WSGI入口
├── app/                         # 核心应用模块
│   ├── models.py                # 数据库模型
│   ├── views.py                 # 视图函数
│   └── urls.py                  # 应用路由
├── user/                        # 用户管理模块
│   ├── models.py                # 用户模型
│   ├── views.py                 # 用户视图
│   └── urls.py                  # 用户路由
├── data/                        # 数据处理管道
│   ├── main.py                  # 管道入口
│   ├── news_spider.py           # 新闻爬虫
│   ├── data_preprocess.py       # 数据预处理
│   ├── predict_sentiment.py     # 情感预测
│   └── news_analysis.py         # 统计分析
├── model/                       # 深度学习模型
│   ├── bilstm_attention.py      # Bi-LSTM+Attention模型
│   ├── train.py                 # 模型训练
│   ├── predict.py               # 模型预测
│   ├── main.py                  # 模型入口
│   ├── checkpoints/             # 模型权重
│   ├── dataset/                 # 数据集
│   └── utils/                   # 工具函数
├── templates/                   # 前端模板
├── static/                      # 静态资源
├── config.py                    # 数据库配置
└── manage.py                    # Django管理命令
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- MySQL 5.7+
- 虚拟环境

### Steps

1. **克隆项目**
```bash
git clone https://github.com/axlumen/news_sentiment_analysis_system.git
cd news_sentiment_analysis_system
```

2. **创建虚拟环境**
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置数据库**

编辑 `config.py`：
```python
host = '127.0.0.1'
port = 3306
user = 'root'
password = '123456'
database = 'news_sentiment'
```

5. **执行数据库迁移**
```bash
python manage.py makemigrations
python manage.py migrate
```

6. **创建超级用户**
```bash
python manage.py createsuperuser
```

7. **启动服务**
```bash
python manage.py runserver 0.0.0.0:8000
```

8. **访问应用**

打开浏览器访问 `http://localhost:8000`

## 📊 Usage

### 运行数据处理管道

```bash
# 完整流程（爬虫→预处理→预测→分析）
python data/main.py

# 跳过某些步骤
python data/main.py --skip-spider      # 跳过爬虫
python data/main.py --skip-preprocess  # 跳过预处理
python data/main.py --skip-predict     # 跳过预测
python data/main.py --skip-analysis    # 跳过分析
```

### 模型训练

```bash
cd model

# 数据预处理（生成train/val/test）
python main.py init_data

# 构建词表
python main.py preprocess

# 训练模型
python main.py train

# 测试模型（生成分类报告和混淆矩阵）
python main.py predict

# 绘制训练曲线
python main.py plot_curve
```

## 🔌 API Endpoints

| 接口 | 方法 | 功能 |
|------|------|------|
| `/api/analyze_text/` | POST | 单文本情感分析 |
| `/api/overall_sentiment/` | GET | 整体情感统计 |
| `/api/daily_sentiment/` | GET | 每日情感趋势 |
| `/api/category_sentiment/` | GET | 分类情感统计 |
| `/api/source_sentiment/` | GET | 来源情感统计 |
| `/api/score_distribution/` | GET | 得分分布 |
| `/api/sentiment_keywords/` | GET | 关键词词云数据 |
| `/api/news_list/` | GET | 新闻列表 |
| `/api/run_pipeline/` | POST | 运行数据全流程 |

### 文本情感分析示例

**请求**：
```json
POST /api/analyze_text/
{
    "text": "今日A股三大指数集体上涨，市场情绪积极。"
}
```

**响应**：
```json
{
    "sentiment": "positive",
    "score": 0.8523
}
```

## 🧠 Model Architecture

```
输入序列 → 词嵌入层 → Bi-LSTM → Attention → 全连接层 → 输出Logits
    ↓             ↓          ↓           ↓
  Token ID     128维       2×256维      加权池化    3分类
```

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | 动态 | 基于训练集构建 |
| embedding_dim | 128 | 词嵌入维度 |
| hidden_dim | 256 | LSTM隐藏层维度 |
| n_layers | 2 | LSTM层数 |
| output_dim | 3 | 三分类输出 |
| dropout | 0.5 | Dropout比例 |

## 📈 Model Performance

| 指标 | 值 |
|------|-----|
| 准确率 (Accuracy) | 92% |
| 精确率 (Precision) | 91% |
| 召回率 (Recall) | 92% |
| F1 分数 | 91% |

## 📁 Database Tables

| 表名 | 用途 |
|------|------|
| `app_newsraw` | 原始新闻数据 |
| `app_newssentimentanalysis` | 情感分析结果 |
| `overall_sentiment` | 整体情感统计 |
| `source_sentiment` | 来源情感统计 |
| `category_sentiment` | 分类情感统计 |
| `date_sentiment` | 每日情感趋势 |
| `score_distribution` | 得分分布 |
| `sentiment_keywords` | 情感关键词 |

## 🛡 Security Notes

- 生产环境修改 `SECRET_KEY`
- 关闭 `DEBUG` 模式
- 配置正确的 `ALLOWED_HOSTS`
- 保护数据库密码（建议使用环境变量）
- 爬虫延迟控制（2-4秒）

## 🤝 Contributing

欢迎提交 Issue 和 Pull Request！

## 📄 License

This project is licensed under the MIT License.

## 📞 Contact

如有问题，请查看项目日志或联系开发者。
