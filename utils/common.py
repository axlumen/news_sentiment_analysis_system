"""通用工具函数模块"""
import logging
import time
import hashlib
from functools import wraps
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def timing_decorator(func):
    """
    函数执行时间统计装饰器
    
    记录函数执行耗时，便于性能分析
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.debug(f"函数 {func.__name__} 执行耗时: {duration:.4f} 秒")
        return result
    return wrapper


def generate_hash(data: str) -> str:
    """
    生成字符串的MD5哈希值
    
    参数：
        data: 输入字符串
    
    返回：
        MD5哈希值（32位十六进制）
    """
    return hashlib.md5(data.encode('utf-8')).hexdigest()


def safe_get(dict_obj: Dict, key: str, default: Any = None) -> Any:
    """
    安全获取字典值，避免KeyError
    
    参数：
        dict_obj: 字典对象
        key: 键名
        default: 默认值
    
    返回：
        字典值或默认值
    """
    if isinstance(dict_obj, dict):
        return dict_obj.get(key, default)
    return default


def chunk_list(items: List, chunk_size: int) -> List[List]:
    """
    将列表分割成指定大小的块
    
    参数：
        items: 原始列表
        chunk_size: 每个块的大小
    
    返回：
        分割后的块列表
    """
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def remove_none_values(dict_obj: Dict) -> Dict:
    """
    移除字典中值为None的键值对
    
    参数：
        dict_obj: 字典对象
    
    返回：
        过滤后的字典
    """
    return {k: v for k, v in dict_obj.items() if v is not None}


def format_datetime(dt) -> Optional[str]:
    """
    格式化日期时间对象为字符串
    
    参数：
        dt: 日期时间对象
    
    返回：
        ISO格式的日期时间字符串或None
    """
    if dt is None:
        return None
    return dt.isoformat()


def truncate_string(text: str, max_length: int = 100) -> str:
    """
    截断字符串到指定长度
    
    参数：
        text: 原始字符串
        max_length: 最大长度
    
    返回：
        截断后的字符串（超过长度时添加省略号）
    """
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length] + '...'


def parse_int(value, default: int = 0) -> int:
    """
    安全解析整数
    
    参数：
        value: 待解析的值
        default: 默认值
    
    返回：
        解析后的整数或默认值
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def parse_float(value, default: float = 0.0) -> float:
    """
    安全解析浮点数
    
    参数：
        value: 待解析的值
        default: 默认值
    
    返回：
        解析后的浮点数或默认值
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def ensure_list(value) -> List:
    """
    确保值为列表类型
    
    参数：
        value: 任意值
    
    返回：
        列表类型的值
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]