import sys
import os
import getpass
import platform
import logging
import yaml

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # 系统环境
ARM64 = platform.machine() in {"arm64", "aarch64"}  # ARM64

if sys.version_info < (3, 10):
    raise RuntimeError(f"需要最低python版本3.10。当前版本为{sys.version_info}")

def load_cfg(yaml_path='cfg/config.yaml') -> dict:
    """
    从配置文件中读取所有配置和设置
    returns:
        dict: 包含所有配置信息的字典
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        print(f"程序初始化失败：读取配置文件失败: {str(e)}")
        sys.exit(0)

CONFIG_AND_SETTINGS = load_cfg()
SERVER_CONFIG = load_cfg('cfg/server_config.yaml')

def set_logging():
    """
    以UTF-8编码创建logger
    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            match record.levelno:
                case logging.INFO:
                    record.msg = f"🤖\u3000{record.msg}"
                case logging.WARNING:
                    record.msg = f"⚠️\u3000{record.msg}"
                case logging.ERROR:
                    record.msg = f"❌\u3000{record.msg}"
                case logging.CRITICAL:
                    record.msg = f"🤯\u3000{record.msg}" 
                case logging.DEBUG:
                    record.msg = f"👾\u3000{record.msg}"
            return super().format(record)

    # 适配Windows系统
    formatter = CustomFormatter("%(message)s")  # Default formatter
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        try:
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        except Exception as e:
            print(f"Warning: Creating SafeFormatter for non UTF-8 environments due to {e}")
            class SafeFormatter(logging.Formatter):
                def format(self, record):
                    """Format log records with UTF-8 encoding for Windows compatibility."""
                    return super().format(record).encode().decode("ascii", "ignore") if WINDOWS else super().format(record)
            formatter = SafeFormatter("%(message)s")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)

    logger = logging.getLogger('QWenIA')
    logger.setLevel(CONFIG_AND_SETTINGS['log_level'])
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger

LOGGER = set_logging()

def colorstr(*input):
    r"""
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.
    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')
    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.
    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'
    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]

PRIFIX = colorstr("QwenIA: ")

if WINDOWS:
    username = getpass.getuser()
    CACHE_DIR = f'C:/Users/{username}/.iaCache'
elif LINUX:
    CACHE_DIR = '/root/.iaCache'
elif MACOS:
    username = getpass.getuser()
    CACHE_DIR = f'/Users/{username}/.iaCache'
else:
    LOGGER.warning(
        f"Unsupported operating system: {platform.system()}. Defaulting cache to current working directory."
    )
    CACHE_DIR = os.path.join(os.getcwd(), '.iaCache')

# Debug Only
CACHE_DIR = './.iaCache'