from django.db import models
from django.contrib.auth.models import AbstractUser

class CustomUser(AbstractUser):
    """自定义用户模型"""
    username = models.CharField('用户名', max_length=100, unique=True)
    password = models.CharField('密码', max_length=100)
    address = models.CharField('地址', max_length=100, blank=True, null=True)
    avatar = models.ImageField('头像', upload_to='avatars/', blank=True, null=True)
    is_admin = models.BooleanField('是否为管理员', default=False)

    phone = models.CharField('电话号码', max_length=20, blank=True)
    
    def __str__(self):
        return self.username
    
    class Meta:
        verbose_name = '用户管理'
        verbose_name_plural = '用户管理'