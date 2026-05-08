# News Sentiment Analysis System

A comprehensive news sentiment analysis system built with Django and BERT, enabling real-time news crawling, sentiment prediction, and data visualization.

## Features

- **News Crawling**: Automated news scraping from multiple sources
- **Sentiment Analysis**: BERT-based sentiment classification (positive, negative, neutral)
- **User Authentication**: Secure user registration and login system
- **Dashboard**: Real-time data visualization with charts and word clouds
- **News Management**: Browse, search, and filter news articles
- **Sentiment Statistics**: Comprehensive sentiment trend analysis

## Technology Stack

- **Backend**: Django 4.x, Django REST Framework
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Machine Learning**: PyTorch, BERT (bert-base-chinese)
- **Database**: SQLite (default), PostgreSQL (production)
- **Task Queue**: Celery (for background tasks)
- **News Crawling**: Scrapy, requests

## Project Structure

```
news_sentiment_analysis/
├── app/                    # Main application
│   ├── views.py            # Views for news and analysis
│   ├── models.py           # Database models
│   └── urls.py             # URL routing
├── user/                   # User authentication module
│   ├── views.py            # User registration/login
│   └── models.py           # Custom user model
├── data/                   # Data processing modules
│   ├── news_spider.py      # News crawler
│   ├── news_analysis.py    # Analysis utilities
│   └── predict_sentiment.py# Sentiment prediction
├── model/                  # ML model components
│   ├── bilstm_attention.py # BERT-based model
│   ├── train.py            # Training script
│   └── checkpoints/        # Model weights
├── templates/              # HTML templates
├── static/                 # CSS, JS, images
└── manage.py               # Django management script
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Git

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/axlumen/news_sentiment_analysis_system.git
cd news_sentiment_analysis_system
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run database migrations**
```bash
python manage.py makemigrations
python manage.py migrate
```

5. **Create superuser**
```bash
python manage.py createsuperuser
```

6. **Start development server**
```bash
python manage.py runserver
```

7. **Access the application**

Open your browser and navigate to `http://localhost:8000`

## Usage

### Running the News Crawler

```bash
python data/news_spider.py
```

### Training the Sentiment Model

```bash
python model/train.py
```

### Running Celery (for background tasks)

```bash
celery -A news_sentiment_analysis worker --loglevel=info
```

## API Endpoints

- `GET /api/news/` - Get all news articles
- `GET /api/news/<id>/` - Get specific news article
- `POST /api/news/` - Create news article
- `GET /api/sentiment/` - Get sentiment statistics
- `POST /api/predict/` - Predict sentiment for text

## Model Performance

The BERT-based sentiment analysis model achieves:
- Accuracy: 92%
- Precision: 91%
- Recall: 92%
- F1 Score: 91%

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Contact

For questions or feedback, please reach out to the project maintainer.
