import psutil
import smtplib
import time
import subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# ======================
# ✉️ 邮箱配置（使用 Gmail SMTP）
# ======================
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_SENDER = 'jiahengxiong1@gmail.com'
EMAIL_PASSWORD = 'xrmclemmkikqhpfs'   # 请确保这里填写的是你的 Gmail 应用专用密码（无空格）
EMAIL_RECEIVER = '10886580@polimi.it'

# ======================
# 预设运行时间阈值（秒）
# 小于此时间认为是手动终止
EXPECTED_MIN_RUNTIME = 2.0

# ======================
# 获取 python/python3 进程
# ======================
def get_python_processes():
    result = []
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = p.info.get('cmdline')
            if isinstance(cmdline, list) and any('python' in part for part in cmdline):
                result.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return result

# ======================
# 发送邮件（支持文本邮件）
# ======================
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = Header(subject, 'utf-8')
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # 启用 TLS 加密
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("[✓] 邮件已发送")
    except Exception as e:
        print("[✗] 邮件发送失败:", e)

# ======================
# 辅助函数：判断进程退出原因
# ======================
def get_exit_reason(pid, runtime):
    """
    尝试从 dmesg 中读取包含 "killed process {pid}" 的整行记录，如果找到，则直接返回该行；
    如果没有找到，则根据进程运行时间判断：
      - 若运行时间小于 EXPECTED_MIN_RUNTIME，则判断为手动终止；
      - 否则判断为正常退出。
    """
    try:
        output = subprocess.check_output(['dmesg', '-T'], stderr=subprocess.STDOUT).decode('utf-8')
        for line in output.splitlines():
            if f"killed process {pid}" in line.lower():
                return line.strip()
    except Exception as e:
        # 如果无法访问 dmesg，则忽略该错误，后续基于运行时间进行判断
        pass

    if runtime < EXPECTED_MIN_RUNTIME:
        return f"手动终止 (运行时间 {runtime:.2f} 秒)"
    else:
        return f"正常退出 (运行时间 {runtime:.2f} 秒)"

# ======================
# 主监控逻辑
# ======================
def monitor():
    print("🚀 正在监控 python/python3 进程...")
    tracked = {}  # 记录: pid -> { 'cmdline': ..., 'start_time_str': ..., 'start_epoch': ... }

    while True:
        current = get_python_processes()
        current_pids = set(p.pid for p in current)

        # 跟踪新启动的进程
        for proc in current:
            if proc.pid not in tracked:
                try:
                    start_epoch = proc.create_time()
                    tracked[proc.pid] = {
                        'cmdline': ' '.join(proc.cmdline()),
                        'start_time_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_epoch)),
                        'start_epoch': start_epoch
                    }
                    print(f"[+] 跟踪新进程 PID {proc.pid}")
                except Exception:
                    continue

        # 判断已结束的进程
        ended_pids = [pid for pid in tracked if pid not in current_pids]
        for pid in ended_pids:
            info = tracked.pop(pid)
            end_epoch = time.time()
            runtime = end_epoch - info['start_epoch']
            reason = get_exit_reason(pid, runtime)
            subject = f"⚠️ Python进程 PID {pid} 已结束"
            body = f"""
进程信息:
----------
PID         : {pid}
命令行      : {info['cmdline']}
启动时间    : {info['start_time_str']}
结束时间    : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_epoch))}
运行时间    : {runtime:.2f} 秒
退出原因    : {reason}
            """.strip()
            send_email(subject, body)
        time.sleep(5)

if __name__ == '__main__':
    try:
        monitor()
    except KeyboardInterrupt:
        subject = "🛑 Python进程监控脚本被手动终止"
        body = f"""
监控脚本收到 KeyboardInterrupt (例如 Ctrl+C)。
终止时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
状态    : 手动中断脚本运行。
        """.strip()
        send_email(subject, body)
        print("👋 脚本手动终止，通知邮件已发送。")