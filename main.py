import logging
import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from utils.cli_utils import main_opt_get, main_read_config
from utils.config_utils import instantiate_from_config
from utils.file_utils import MainFileManage
from utils.txt_logger import TextFileLogger

class DualOutput:
    """同时输出到屏幕和文件的代理类（无需修改原有print逻辑）"""
    def __init__(self, filename):
        self.terminal = sys.stdout  # 原始屏幕输出
        self.log = open(filename, "a", buffering=1, encoding='utf-8')  # 文件输出
    def write(self, message):
        # 同时写入屏幕和文件
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        # 确保实时刷新
        self.terminal.flush()
        self.log.flush()

if __name__ == '__main__':
    # Parse Command Line Arguments.
    opt = main_opt_get()
    base_config = main_read_config(opt)
    # Instantiate dataloader and model.
    if opt.batch_size is not None:
        base_config['data']['params']['batch_size'] = opt.batch_size
    if opt.learning_rate is not None:
        base_config['model']['params']['learning_rate'] = opt.learning_rate

    batch_size = base_config['data']['params']['batch_size']
    learning_rate = ("{:.0e}".format(base_config['model']['params']['learning_rate'])
                     .replace("e-0", "e-").replace("e+0", "e+"))
    save_name = opt.name
    if opt.model is not None:
        base_config['model']['params']['model'] = opt.model
        save_name = save_name + f"_M{opt.model}"
    save_name = save_name + f"_bc{batch_size}_lr{learning_rate}"
    if opt.drop_out is not None:
        base_config['model']['params']['drop_out'] = opt.drop_out
        drop_out = opt.drop_out
        save_name = save_name + f"_drop0{int(drop_out*10)}"

    pl_dataset = instantiate_from_config(base_config['data'])
    model = instantiate_from_config(base_config['model'])
    # Configure training settings.

    log_setting = MainFileManage(opt.config_name, save_name, base_config, resume_model=opt.resume)
    logger = TensorBoardLogger(log_setting.save_tensorboard_log_path, name=opt.name)

    log_path = os.path.join(log_setting.save_logstream_path, 'stream.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # 重定向标准输出和错误输出
    sys.stdout = DualOutput(log_path)
    sys.stderr = DualOutput(log_path)  # 可选：同时捕获错误输出

    if opt.log:
        enable_progress_bar = False
    else:
        enable_progress_bar = True

    csv_logger = CSVLogger(save_dir=log_setting.save_metrics_path)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控验证损失
        dirpath=log_setting.save_checkpoints_path,  # 模型保存的目录
        filename='{epoch}-{val_loss:.2f}',  # 文件命名模板
        save_top_k=1,  # 只保存最好的模型
        mode='min',  # 最小化 val_loss
        save_weights_only=True  # 只保存模型权重
    )
    lightning_setting = base_config['lightning']
    trainer = Trainer(**lightning_setting['lightning']['trainer'], logger=[logger, csv_logger],
                      callbacks=[checkpoint_callback], devices=opt.gpus, max_epochs=opt.epoch,
                      enable_progress_bar=enable_progress_bar)
    # Begin training or testing
    if not opt.test:
        trainer.fit(model, pl_dataset, ckpt_path=opt.resume)
    else:
        trainer.test(model, pl_dataset, ckpt_path=opt.resume)