import sys

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
import os

class TextFileLogger(Logger):
    def __init__(self, log_file='training_log.txt'):
        super().__init__()
        self.log_file = log_file
        self._setup_file()

    def _setup_file(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'a') as f:
            f.write('Training Log\n')

    @rank_zero_only  # 确保只在主进程记录（分布式训练时避免重复写入）
    def log_metrics(self, metrics, step):
        log_str = f"Step {step} - " + " | ".join([f"{k}: {v}" for k, v in metrics.items()])
        with open(self.log_file, 'a') as f:
            f.write(log_str + '\n')

    @property
    def name(self):
        return "TextLogger"

    @property
    def version(self):
        return "1.0"


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout  # 保存原始标准输出
        self.log = open(log_file, "a", encoding="utf-8")  # 追加写入日志文件

    def write(self, message):
        self.terminal.write(message)  # 输出到终端
        self.log.write(message)       # 写入日志文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()  # 对象销毁时关闭文件