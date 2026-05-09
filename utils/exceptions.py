"""统一异常处理模块"""
import json
import logging

from django.http import JsonResponse
from functools import wraps

logger = logging.getLogger(__name__)


class AppError(Exception):
    """自定义应用异常"""
    def __init__(self, message, code=400, details=None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ValidationError(AppError):
    """数据验证异常"""
    def __init__(self, message, field=None):
        super().__init__(message, code=400)
        self.field = field


class ResourceNotFoundError(AppError):
    """资源未找到异常"""
    def __init__(self, message):
        super().__init__(message, code=404)


class AuthenticationError(AppError):
    """认证异常"""
    def __init__(self, message):
        super().__init__(message, code=401)


class PermissionError(AppError):
    """权限异常"""
    def __init__(self, message):
        super().__init__(message, code=403)


def api_response(data=None, message='success', code=200, **kwargs):
    """
    统一API响应格式
    
    参数：
        data: 响应数据
        message: 响应消息
        code: HTTP状态码
        **kwargs: 其他参数
    
    返回：
        JsonResponse
    """
    response = {
        'code': code,
        'message': message,
        'data': data,
        **kwargs
    }
    return JsonResponse(response, status=code, safe=False)


def api_error(message, code=400, details=None):
    """
    统一错误响应格式
    
    参数：
        message: 错误消息
        code: HTTP状态码
        details: 错误详情
    
    返回：
        JsonResponse
    """
    response = {
        'code': code,
        'message': message,
        'data': None,
        'details': details or {}
    }
    return JsonResponse(response, status=code)


def handle_exceptions(func):
    """
    API视图异常处理装饰器
    
    捕获并处理所有异常，返回统一的错误响应格式
    """
    @wraps(func)
    def wrapper(request, *args, **kwargs):
        try:
            return func(request, *args, **kwargs)
        except AppError as e:
            logger.error(f"应用异常: {e.message}")
            return api_error(e.message, e.code, e.details)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}")
            return api_error(f'JSON解析错误: {str(e)}', 400)
        except Exception as e:
            logger.error(f"未处理异常: {str(e)}", exc_info=True)
            return api_error(f'服务器内部错误: {str(e)}', 500)
    return wrapper


def validate_request_params(required_params):
    """
    请求参数验证装饰器
    
    参数：
        required_params: 必填参数列表
    
    返回：
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            if request.method == 'POST':
                try:
                    data = json.loads(request.body) if request.body else {}
                except json.JSONDecodeError:
                    return api_error('无效的JSON数据', 400)
            else:
                data = request.GET
            
            missing_params = [p for p in required_params if p not in data]
            if missing_params:
                return api_error(f'缺少必要参数: {", ".join(missing_params)}', 400)
            
            return func(request, *args, **kwargs)
        return wrapper
    return decorator