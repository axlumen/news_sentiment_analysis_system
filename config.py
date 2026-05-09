import os
from dotenv import load_dotenv
from typing import bool

load_dotenv()

class Config:
    """项目配置类 - 统一管理所有环境变量"""
    
    # 数据库配置
    host: str = os.getenv('DB_HOST', '127.0.0.1')
    port: int = int(os.getenv('DB_PORT', '3306'))
    user: str = os.getenv('DB_USER', 'root')
    password: str = os.getenv('DB_PASSWORD', '123456')
    database: str = os.getenv('DB_NAME', 'news_sentiment')
    charset: str = os.getenv('DB_CHARSET', 'utf8mb4')
    
    # Django配置
    secret_key: str = os.getenv('SECRET_KEY', '')
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # 模型配置
    model_path: str = os.getenv('MODEL_PATH', 'model/checkpoints/best_model.pth')
    vocab_path: str = os.getenv('VOCAB_PATH', 'model/dataset/processed/vocab.pkl')
    
    # 缓存配置
    cache_timeout: int = int(os.getenv('CACHE_TIMEOUT', '3600'))
    
    @classmethod
    def validate(cls) -> bool:
        """验证关键配置是否存在
        
        检查生产环境必需的配置项，确保系统安全运行。
        
        返回:
            bool: 验证通过返回True
            
        抛出:
            ValueError: 验证失败时抛出，包含错误信息列表
        """
        errors: list[str] = []
        if not cls.secret_key and not cls.debug:
            errors.append('SECRET_KEY must be set in production environment')
        if not cls.database:
            errors.append('DB_NAME must be set')
        if errors:
            raise ValueError('\n'.join(errors))
        return True