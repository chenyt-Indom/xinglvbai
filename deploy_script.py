"""服务器部署脚本（自动SSH密钥认证）— 从 GitHub xinglvbai 仓库拉取部署

用法：python deploy_script.py
流程：git fetch + reset 到 xinglvbai/main → 写入 .env → 安装依赖 → 重启服务 → 重载 Nginx → 健康检查
说明：服务器 /opt/lvbai 是纯部署目录，禁止直接改里面的代码文件（会被 reset 覆盖）；
      线上 HTTPS 的 nginx 配置由 update_nginx.py 管理，本脚本不会覆盖它。
"""
import paramiko, time, sys, os

host = '139.199.69.88'

def run(c, cmd, wait=1):
    print(f'  RUN: {cmd[:80]}...')
    stdin, stdout, stderr = c.exec_command(cmd)
    time.sleep(wait)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if err and 'WARNING' not in err:
        print(f'  ERR: {err[:200]}')
    if out:
        print(f'  OUT: {out[:300]}')
    return out, err

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# 自动SSH密钥认证
key_path = os.path.expanduser('~/.ssh/id_rsa')
key = paramiko.RSAKey.from_private_key_file(key_path)
c.connect(host, username='ubuntu', pkey=key, timeout=15)
print('SSH密钥认证成功\n')

# 1. git 拉取最新代码（服务器 git 已对齐 chenyt-Indom/xinglvbai，
#    reset --hard 保证服务器与 GitHub 完全一致，避免合并产生冲突）
print('1. 更新代码（git fetch + reset 到 xinglvbai/main）...')
run(c, 'git config --global --add safe.directory /opt/lvbai')
run(c, 'cd /opt/lvbai && sudo git fetch origin 2>&1 && sudo git reset --hard origin/main 2>&1')

# 2. 创建 .env（从本地 .env 读取密钥）
print('2. 创建 .env...')
# 读取本地 .env 文件中的密钥
env_path = os.path.join(os.path.dirname(__file__), 'deploy', '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        env_content = f.read()
    # 将 .env 内容写入服务器
    escaped = env_content.replace("'", "'\\''")
    run(c, f"sudo bash -c \"echo '{escaped}' > /opt/lvbai/deploy/.env\"")
    run(c, 'grep -c "=" /opt/lvbai/deploy/.env | xargs echo ".env 键值对数量:"')
else:
    print('  警告：未找到本地 deploy/.env 文件，跳过')

# 3. 安装依赖（已安装的会快速跳过；--break-system-packages 兼容 Ubuntu 23.04+ 的 PEP668 限制）
print('3. 安装 Python 依赖...')
run(c, 'cd /opt/lvbai && sudo pip3 install --break-system-packages -r backend/requirements.txt 2>&1', 5)

# 4. 复制 service 并启动
print('4. 配置并启动服务...')
run(c, 'sudo cp /opt/lvbai/deploy/lvbai.service /etc/systemd/system/')
run(c, 'sudo systemctl daemon-reload')
run(c, 'sudo systemctl enable lvbai 2>&1')
run(c, 'sudo systemctl restart lvbai 2>&1')

# 5. Nginx：只测试并重载，不覆盖线上 HTTPS 配置（SSL 证书在 /etc/letsencrypt，
#    由 update_nginx.py 管理；仓库里的 nginx.conf 仅是模板）
print('5. 重载 Nginx（保留线上 HTTPS 配置）...')
run(c, 'sudo nginx -t 2>&1 && sudo systemctl reload nginx 2>&1')

# 6. 验证
print('6. 验证...')
run(c, 'curl -s http://localhost:8000/api/health')
run(c, 'cd /opt/lvbai && sudo git log --oneline -1')

c.close()
print('\n部署完成！')
