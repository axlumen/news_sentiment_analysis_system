from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.shortcuts import render, redirect
from user.models import CustomUser
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
import os

# 创建自定义用户创建表单，移除密码验证要求
from django import forms
from django.contrib.auth.hashers import make_password, check_password

# 尝试导入验证码库，如果没有安装则跳过
try:
    from captcha.fields import CaptchaField
    from captcha.widgets import CaptchaTextInput
    CAPTCHA_AVAILABLE = True
except ImportError:
    CAPTCHA_AVAILABLE = False

class CustomUserCreationForm(forms.ModelForm):
    password1 = forms.CharField(
        label='密码',
        widget=forms.PasswordInput,
        required=True
    )
    password2 = forms.CharField(
        label='确认密码',
        widget=forms.PasswordInput,
        required=True
    )
    email = forms.EmailField(
        label='电子邮箱',
        required=False
    )
    phone = forms.CharField(
        label='电话号码',
        required=False,
        max_length=20
    )
    
    # 添加验证码字段
    if CAPTCHA_AVAILABLE:
        captcha = CaptchaField(label='验证码')
    
    class Meta:
        model = CustomUser
        fields = ('username', 'email', 'phone', 'password1', 'password2') + (('captcha',) if CAPTCHA_AVAILABLE else ())
    
    def clean_password2(self):
        # 只验证密码是否一致，不验证密码强度
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError('两次输入的密码不一致')
        return password2
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.password = make_password(self.cleaned_data['password1'])
        if commit:
            user.save()
        return user

# Create your views here.
# 创建自定义登录表单，支持验证码
class CustomAuthenticationForm(AuthenticationForm):
    if CAPTCHA_AVAILABLE:
        captcha = CaptchaField(label='验证码')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 移除默认的错误信息格式
        if hasattr(self, 'error_messages'):
            self.error_messages['invalid_login'] = '用户名或密码错误'

# 创建用户信息修改表单
class UserProfileForm(forms.ModelForm):
    class Meta:
        model = CustomUser
        fields = ('email', 'address', 'phone')

# 创建密码修改表单
class PasswordChangeForm(forms.Form):
    old_password = forms.CharField(
        label='原密码',
        widget=forms.PasswordInput,
        required=True
    )
    new_password1 = forms.CharField(
        label='新密码',
        widget=forms.PasswordInput,
        required=True
    )
    new_password2 = forms.CharField(
        label='确认新密码',
        widget=forms.PasswordInput,
        required=True
    )
    
    def clean_old_password(self):
        old_password = self.cleaned_data.get('old_password')
        if not check_password(old_password, self.user.password):
            raise forms.ValidationError('原密码错误')
        return old_password
    
    def clean_new_password2(self):
        new_password1 = self.cleaned_data.get('new_password1')
        new_password2 = self.cleaned_data.get('new_password2')
        if new_password1 and new_password2 and new_password1 != new_password2:
            raise forms.ValidationError('两次输入的新密码不一致')
        return new_password2
    
    def __init__(self, user, *args, **kwargs):
        self.user = user
        super().__init__(*args, **kwargs)


def login_view(request):
    """
    用户登录视图
    """
    if request.method == 'POST':
        form_class = CustomAuthenticationForm if CAPTCHA_AVAILABLE else AuthenticationForm
        form = form_class(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            remember = request.POST.get('remember', False)

            # 验证用户
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)

                # 处理记住密码
                if remember:
                    request.session.set_expiry(604800)  # 7天
                else:
                    request.session.set_expiry(0)

                # 移除登录成功提示
                return redirect('app:index')
            else:
                # 添加登录错误信息到表单
                form.add_error(None, '用户名或密码错误')
        # 表单无效时，会自动处理错误信息
    else:
        form_class = CustomAuthenticationForm if CAPTCHA_AVAILABLE else AuthenticationForm
        form = form_class()
    return render(request, 'auth/login.html', {'form': form, 'captcha_available': CAPTCHA_AVAILABLE})

def register_view(request):
    """
    用户注册视图
    """
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        # 检查用户名是否已存在
        username = request.POST.get('username')
        if CustomUser.objects.filter(username=username).exists():
            messages.error(request, '用户名已存在')
            return render(request, 'auth/register.html', {'form': form})
        
        if form.is_valid():
            try:
                user = form.save()
                
                # 处理头像上传
                if 'avatar' in request.FILES:
                    avatar = request.FILES['avatar']
                    
                    # 验证图片格式
                    allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico', '.tif', '.tiff']
                    ext = os.path.splitext(avatar.name)[1].lower()
                    if ext not in allowed_extensions:
                        messages.error(request, '只支持JPG、JPEG、PNG、GIF、BMP、WebP、SVG、ICO、TIFF格式的图片')
                        return redirect('user:register')
                    
                    # 验证图片大小（限制2MB）
                    if avatar.size > 2 * 1024 * 1024:
                        messages.error(request, '图片大小不能超过2MB')
                        return redirect('user:register')
                    
                    # 保存头像
                    fs = FileSystemStorage(location='media/avatars')
                    filename = f'{user.id}_{avatar.name}'
                    # 保存新头像
                    filename = fs.save(filename, avatar)
                    user.avatar = f'avatars/{filename}'
                    user.save()
                
                messages.success(request, '注册成功！请登录')
                return redirect('user:login')
            except Exception as e:
                messages.error(request, f'注册失败: {str(e)}')
        else:
            # 显示具体的表单错误
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = CustomUserCreationForm()
    return render(request, 'auth/register.html', {'form': form})

def logout_view(request):
    """
    用户登出视图
    """
    logout(request)
    return redirect('user:login')

@login_required
def profile_view(request):
    """
    个人中心视图
    """
    import os
    from django.core.files.storage import FileSystemStorage
    
    user = request.user
    if request.method == 'POST':
        # 处理表单数据
        form = UserProfileForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            
            # 处理头像上传
            if 'avatar' in request.FILES:
                avatar = request.FILES['avatar']
                
                # 验证图片格式
                allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico', '.tif', '.tiff']
                ext = os.path.splitext(avatar.name)[1].lower()
                if ext not in allowed_extensions:
                    messages.error(request, '只支持JPG、JPEG、PNG、GIF、BMP、WebP、SVG、ICO、TIFF格式的图片')
                    return redirect('user:profile')
                
                # 验证图片大小（限制2MB）
                if avatar.size > 2 * 1024 * 1024:
                    messages.error(request, '图片大小不能超过2MB')
                    return redirect('user:profile')
                
                # 保存头像
                fs = FileSystemStorage(location='media/avatars')
                fs = FileSystemStorage(location='media/avatars')
                filename = f'{request.user.id}_{avatar.name}'
                # 删除旧头像
                if request.user.avatar and os.path.exists(os.path.join('media', str(request.user.avatar))):
                    try:
                        os.remove(os.path.join('media', str(request.user.avatar)))
                    except:
                        pass
                # 保存新头像
                filename = fs.save(filename, avatar)
                request.user.avatar = f'avatars/{filename}'
                request.user.save()
            
            messages.success(request, '个人信息更新成功！')
            return redirect('user:profile')
        else:
            # 显示表单错误
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = UserProfileForm(instance=user)
    return render(request, 'auth/profile.html', {'user': user, 'form': form})

@login_required

def edit_profile_view(request):
    """
    修改个人信息视图
    """
    user = request.user
    if request.method == 'POST':
        form = UserProfileForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, '个人信息更新成功！')
            return redirect('user:profile')
        else:
            # 显示表单错误
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = UserProfileForm(instance=user)
    return render(request, 'auth/edit_profile.html', {'form': form, 'user': user})

@login_required

def upload_avatar_view(request):
    """
    上传头像视图
    """
    if request.method == 'POST' and request.FILES.get('avatar'):
        avatar = request.FILES['avatar']
        
        # 验证图片格式
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico', '.tif', '.tiff']
        ext = os.path.splitext(avatar.name)[1].lower()
        if ext not in allowed_extensions:
            messages.error(request, '只支持JPG、JPEG、PNG、GIF、BMP、WebP、SVG、ICO、TIFF格式的图片')
            return redirect('user:profile')
        
        # 验证图片大小（限制2MB）
        if avatar.size > 2 * 1024 * 1024:
            messages.error(request, '图片大小不能超过2MB')
            return redirect('user:profile')
        
        # 保存头像
        fs = FileSystemStorage(location='media/avatars')
        filename = f'{request.user.id}_{avatar.name}'
        # 删除旧头像
        if request.user.avatar and os.path.exists(os.path.join('media', str(request.user.avatar))):
            try:
                os.remove(os.path.join('media', str(request.user.avatar)))
            except:
                pass
        # 保存新头像
        filename = fs.save(filename, avatar)
        request.user.avatar = f'avatars/{filename}'
        request.user.save()
        
        messages.success(request, '头像上传成功！')
        return redirect('user:profile')
    return redirect('user:profile')

@login_required

def change_password_view(request):
    """
    修改密码视图
    """
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            # 更新密码
            request.user.password = make_password(form.cleaned_data['new_password1'])
            request.user.save()
            messages.success(request, '密码修改成功！请重新登录')
            logout(request)
            return redirect('user:login')
        else:
            # 显示表单错误
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{field}: {error}')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'auth/change_password.html', {'form': form})