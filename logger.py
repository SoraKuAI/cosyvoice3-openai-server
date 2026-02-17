import os
import sys
import logging
import time
from typing import Optional

class CompactFormatter(logging.Formatter):
    """
    vLLM-style log:
    (APIServer pid=72013) 02-18 02:09:30 openai_api_server:62 INFO: Message...
    """
    # 颜色配置
    COLORS = {
        logging.DEBUG: "\033[36m",    # 青色
        logging.INFO: "\033[32m",     # 绿色
        logging.WARNING: "\033[33m",  # 黄色
        logging.ERROR: "\033[31m",    # 红色
        logging.CRITICAL: "\033[1;31m", # 加粗红
    }
    META_COLOR = "\033[90m"      # 灰色 (用于 PID, 时间, 文件名)
    PARAM_VAL_COLOR = "\033[36m" # 青色 (用于数值)
    PARAM_KEY_COLOR = "\033[96m" # 亮青色 (用于键名)
    RESET = "\033[0m"

    def __init__(self, component_name: str = "APIServer"):
        super().__init__()
        self.component_name = component_name
        self.pid = os.getpid()

    def format(self, record: logging.LogRecord) -> str:
        prefix = f"({self.component_name} pid={self.pid})"
        timestamp = time.strftime("%m-%d %H:%M:%S", time.localtime(record.created))
        
        file_info = f"{os.path.splitext(os.path.basename(record.pathname))[0]}:{record.lineno}"
        
        level_color = self.COLORS.get(record.levelno, "")
        level_name = f"{record.levelname}"
        
        message = record.getMessage()
        
        # 定义需要高亮的关键词
        params = [f"{prefix}", "total_dur=", "dur=", "latency=", "rtf=", "total_dur=", "elapsed=", "speech_len="]
        for p in params:
            if p in message:
                # 将 key=value 变成 [亮青]key=[青]value[重置]
                # 这里用一个简单的拆分逻辑处理
                parts = message.split(p)
                new_message = ""
                for i in range(len(parts)-1):
                    # 尝试抓取数值部分（直到空格或字符串末尾）
                    post_val = parts[i+1].split(" ", 1)
                    val = post_val[0]
                    rest = " " + post_val[1] if len(post_val) > 1 else ""
                    new_message += f"{parts[i]}{self.PARAM_KEY_COLOR}{p}{self.PARAM_VAL_COLOR}{val}{self.RESET}{rest}"
                message = new_message

        # 核心格式化输出
        return (
            f"{self.PARAM_KEY_COLOR}{prefix} {self.RESET}{timestamp} {file_info:<16}{self.RESET} "
            f"{level_color}{level_name:<4}{self.RESET}: " # 级别本身也缩窄并保持对齐
            f"{message}"
        )
        
        
def setup_logger(
    name: str,
    component_name: str = "APIServer",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # 传入组件名
    formatter = CompactFormatter(component_name=component_name)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        # 文件日志去掉颜色，只保留文本格式
        fh = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            f"({component_name} pid={os.getpid()}) %(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s",
            datefmt="%m-%d %H:%M:%S"
        )
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger
