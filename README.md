# 新闻情感分析系统

基于 Django + PyTorch 的中文新闻情感分析系统，实现了新闻爬虫、数据预处理、情感分析预测和可视化展示的完整流程。

## 功能特点

- 📰 **新闻爬虫**：支持新浪新闻多板块自动爬取
- 🔍 **情感分析**：基于 Bi-LSTM + Attention 的深度学习模型
- 📊 **可视化仪表盘**：实时展示情感统计、趋势分析、词云等
- 🚀 **RESTful API**：提供完整的 API 接口供外部调用
- 🐳 **Docker 部署**：支持容器化部署

## 技术栈

- **后端框架**: Django 5.2
- **深度学习**: PyTorch 2.3
- **中文分词**: jieba
- **数据库**: MySQL 8.0
- **可视化**: ECharts
- **部署**: Docker + Nginx + Gunicorn

## 项目结构

```
news_sentiment_analysis/
├── app/                 # 核心应用模块
│   ├── models.py        # 数据库模型
│   ├── views.py         # 视图函数
│   └── urls.py          # 路由配置
├── data/                # 数据处理模块
│   ├── news_spider.py   # 新闻爬虫
│   ├── data_preprocess.py # 数据预处理
│   ├── predict_sentiment.py # 情感预测
│   └── news_analysis.py # 统计分析
├── model/               # 模型模块
│   ├── bilstm_attention.py # Bi-LSTM + Attention 模型
│   ├── train.py         # 模型训练
│   ├── predict.py       # 模型预测
│   ├── model_manager.py # 模型管理器(单例)
│   └── checkpoints/     # 模型权重
├── user/                # 用户管理模块
├── utils/               # 工具函数
│   ├── exceptions.py    # 统一异常处理
│   └── common.py        # 通用工具函数
├── static/              # 静态资源
├── templates/           # 模板文件
├── tests/               # 单元测试
├── Dockerfile           # Docker 配置
├── docker-compose.yml   # Docker Compose 配置
├── nginx/               # Nginx 配置
└── requirements.txt     # 依赖列表
```

## 快速开始

### 环境要求

- Python 3.10+
- MySQL 8.0+
- PyTorch 2.3+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置数据库

1. 创建数据库：
```sql
CREATE DATABASE news_sentiment CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. 配置环境变量（修改 `.env` 文件）：
```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=news_sentiment
SECRET_KEY=your_secret_key
```

### 数据库迁移

```bash
python manage.py makemigrations
python manage.py migrate
```

### 启动开发服务器

```bash
python manage.py runserver 0.0.0.0:8000
```

### 使用 Docker 部署

```bash
# 构建并启动容器
docker-compose up -d

# 查看日志
docker-compose logs -f
```

## API 接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/overall/` | GET | 获取整体情感统计 |
| `/api/date/` | GET | 获取日期情感趋势 |
| `/api/category/` | GET | 获取分类情感统计 |
| `/api/source/` | GET | 获取来源情感统计 |
| `/api/keywords/` | GET | 获取情感关键词 |
| `/api/analyze/` | POST | 分析文本情感 |

### 示例请求

```bash
# 获取整体情感统计
curl http://localhost:8000/api/overall/

# 分析文本情感
curl -X POST http://localhost:8000/api/analyze/ \
  -H "Content-Type: application/json" \
  -d '{"text": "今日A股三大指数集体上涨"}'
```

## 模型训练

```bash
cd model
python main.py train
```

## 项目亮点

1. **安全配置**：环境变量管理敏感信息，生产环境安全加固
2. **性能优化**：缓存机制、数据库索引、模型单例化
3. **代码质量**：统一异常处理、类型提示、单元测试
4. **可扩展性**：模块化设计、RESTful API、Docker 部署

## 许可证

MIT License