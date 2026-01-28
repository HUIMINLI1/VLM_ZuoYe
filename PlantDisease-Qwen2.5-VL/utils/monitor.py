'''
$lhm 251023
性能与服务器状态监视器
'''
import sys, os
import psutil
import time
import threading
import requests
import itertools
from pathlib import Path
from functools import wraps
from utils import LOGGER, SERVER_CONFIG, WINDOWS, LINUX

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

if WINDOWS:
    import wmi, pythoncom

def performance_monitor(path=SERVER_CONFIG['MODEL_PATH'], interval=1.0):
    """
    装饰器：监控指定路径所在磁盘、GPU显存和系统内存使用情况。
    - 若磁盘 busy 时间 >90% 且持续 5s，发出警告
    - 若显存 >95%，发出警告
    - 若系统内存 >90%，发出警告
    args:
        path: 默认为Qwen模型所在路径，监测的磁盘是path所在的磁盘。
        interval: 监测间隔（单位：秒）。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            stop_flag = threading.Event()

            warning_state = {
                'disk': False,
                'memory': False,
                'gpu': False
            }

            disk_over_threshold_secs = 0

            # 查找挂载路径
            mount_path = Path(path).resolve()
            while not os.path.ismount(str(mount_path)) and mount_path != mount_path.parent:
                mount_path = mount_path.parent

            target_partition = None
            for part in psutil.disk_partitions():
                try:
                    if os.path.samefile(part.mountpoint, str(mount_path)):
                        target_partition = part.device
                        break
                except Exception as e:
                    LOGGER.error(f"性能监控已关闭：{e}")
                    return func(*args, **kwargs)
            else:
                LOGGER.error("性能监控已关闭：未找到目标路径的挂载分区。")
                return func(*args, **kwargs)
                
            def monitor():
                nonlocal disk_over_threshold_secs

                if WINDOWS:
                    pythoncom.CoInitialize()
                    w = wmi.WMI(namespace="root\\CIMV2")
                    logical_drive = os.path.splitdrive(str(mount_path))[0]
                else:
                    part_name = os.path.basename(target_partition)
                    io_prev = psutil.disk_io_counters(perdisk=True).get(part_name)
                    time_prev = time.time()

                reset_interval = 60
                last_reset_time = time.time()

                try:
                    while not stop_flag.is_set():
                        time.sleep(interval)
                        now = time.time()

                        # === 磁盘 busy 百分比 ===
                        if WINDOWS:
                            busy_percent = 0
                            try:
                                for disk in w.Win32_PerfFormattedData_PerfDisk_LogicalDisk():
                                    if disk.Name.upper() == logical_drive.upper():
                                        busy_percent = float(disk.PercentDiskTime) / 100.0
                                        break
                            except Exception as e:
                                LOGGER.warning(f"WMI 查询磁盘使用率失败: {e}")
                        else:
                            elapsed = now - time_prev
                            io_curr = psutil.disk_io_counters(perdisk=True).get(part_name)
                            if io_prev and io_curr:
                                busy_time_delta = getattr(io_curr, "busy_time", 0) - getattr(io_prev, "busy_time", 0)
                                busy_percent = busy_time_delta / (elapsed * 1000) if elapsed > 0 else 0
                                io_prev = io_curr
                                time_prev = now
                            else:
                                busy_percent = 0

                        # === 磁盘警告逻辑 ===
                        if busy_percent > 0.9:
                            disk_over_threshold_secs += interval
                            if disk_over_threshold_secs >= 5 and not warning_state['disk']:
                                LOGGER.warning("监测到性能瓶颈：磁盘IO。请耐心等待。")
                                warning_state['disk'] = True
                        else:
                            disk_over_threshold_secs = 0

                        # === 内存监控 ===
                        mem = psutil.virtual_memory()
                        if mem.percent > 90 and not warning_state['memory']:
                            LOGGER.warning(f"监测到性能瓶颈：内存。占用已达 {mem.percent:.1f}%，请注意资源消耗。")
                            warning_state['memory'] = True

                        # === 显存监控 ===
                        if GPU_AVAILABLE:
                            try:
                                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                                gpu_usage = mem_info.used / mem_info.total
                                if gpu_usage > 0.95 and not warning_state['gpu']:
                                    LOGGER.warning(f"监测到性能瓶颈：显存。占用已达 {gpu_usage:.1%}，请注意资源消耗。")
                                    warning_state['gpu'] = True
                            except Exception as e:
                                LOGGER.warning(f"显存监控异常: {e}")

                        # === 告警重置逻辑 ===
                        if now - last_reset_time >= reset_interval:
                            warning_state['disk'] = False
                            warning_state['memory'] = False
                            warning_state['gpu'] = False
                            last_reset_time = now

                except Exception as e:
                    LOGGER.warning(f"监控线程异常中断: {e}")
                finally:
                    if WINDOWS:
                        pythoncom.CoUninitialize()

            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()

            try:
                return func(*args, **kwargs)
            finally:
                stop_flag.set()
                monitor_thread.join()

        return wrapper
    return decorator


def wait_for_server(port=SERVER_CONFIG['PORT'], interval=1):
    """
    每隔 interval 秒向 health_url 发送一次请求，直到服务器状态为 200。
    - 状态码 200："status": "ok"，表示服务器准备就绪。
    - 状态码 503："message": "Loading model", "type": "unavailable_error"，继续等待。
    - 请求异常或连接错误：打印“等待server启动···”，继续等待。
    """
    url=f"http://localhost:{port}/health"
    spinner = itertools.cycle(['·', '··', '···'])

    while True:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print()
                break
            elif response.status_code == 503:
                dots = next(spinner)
                sys.stdout.write(f"\rllama-server正在初始化{dots}   ")
                sys.stdout.flush()
            else: 
                dots = next(spinner)
                sys.stdout.write(f"\rllama-server连接异常：{response.status_code}{dots}   ")
                sys.stdout.flush()

        except requests.exceptions.RequestException as e:
            if "No connection adapters" in str(e):
                LOGGER.critical(f"无法连接到llama-server：{e}。请检查本地连接代理或其他网络设置。")
                quit(1)
            dots = next(spinner)
            sys.stdout.write(f"\r等待llama-server启动{dots}   ")
            sys.stdout.flush()

        time.sleep(interval)
    print()