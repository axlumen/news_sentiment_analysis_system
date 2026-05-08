/**
 * 登录页面JavaScript
 */
document.addEventListener('DOMContentLoaded', function() {
  // 错误提示自动消失
  const errorAlert = document.querySelector('.error-alert');
  if (errorAlert) {
    setTimeout(function() {
      errorAlert.style.animation = 'slideUp 0.3s ease-in forwards';
      setTimeout(() => errorAlert.remove(), 300);
    }, 3000);
  }

  // 密码显隐切换 - 增加健壮性判断
  const togglePassword = document.getElementById('togglePassword');
  const passwordInput = document.getElementById('password');
  if (togglePassword && passwordInput) {
    togglePassword.addEventListener('click', function() {
      const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
      passwordInput.setAttribute('type', type);
      const icon = this.querySelector('i');
      icon.classList.toggle('fa-eye');
      icon.classList.toggle('fa-eye-slash');
    });
  }

  // 表单提交验证 - 精简逻辑
  const loginForm = document.getElementById('loginForm');
  const loginBtn = document.getElementById('loginBtn');
  if (loginForm && loginBtn) {
    loginForm.addEventListener('submit', function(e) {
      const username = document.getElementById('username')?.value.trim() || '';
      const password = document.getElementById('password')?.value.trim() || '';

      if (!username || !password) {
        e.preventDefault();
        // 移除已存在的错误提示
        const existingError = loginForm.querySelector('.error-message');
        if (existingError) existingError.remove();

        // 创建错误提示
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = '<i class="fas fa-exclamation-circle"></i><span>请填写用户名和密码</span>';
        loginForm.insertBefore(errorDiv, loginForm.firstChild);

        // 3秒后移除
        setTimeout(() => {
          errorDiv.style.opacity = '0';
          setTimeout(() => errorDiv.remove(), 300);
        }, 3000);
        return;
      }

      // 加载状态
      loginBtn.disabled = true;
      loginBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>登录中...';
    });
  }
});

/**
 * 验证码刷新功能
 */
function refreshCaptcha() {
  const captchaImage = document.querySelector('.captcha-image img');
  if (captchaImage) {
    const currentSrc = captchaImage.src;
    const url = new URL(currentSrc);
    url.searchParams.set('t', new Date().getTime());
    captchaImage.src = url.toString();
  }
}