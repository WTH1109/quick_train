from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils.cli_utils import main_opt_get, main_read_config
from utils.config_utils import instantiate_from_config
from utils.file_utils import MainFileManage

if __name__ == '__main__':
    # Parse Command Line Arguments.
    opt = main_opt_get()
    base_config = main_read_config(opt)
    # Instantiate dataloader and model.
    pl_dataset = instantiate_from_config(base_config['data'])
    model = instantiate_from_config(base_config['model'])
    # Configure training settings.
    log_setting = MainFileManage(opt.config_name, opt.name, base_config, resume_model=opt.resume)
    logger = TensorBoardLogger(log_setting.save_tensorboard_log_path, name=opt.name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # 监控验证损失
        dirpath=log_setting.save_checkpoints_path,  # 模型保存的目录
        filename='{epoch}-{val_loss:.2f}',  # 文件命名模板
        save_top_k=1,  # 只保存最好的模型
        mode='min',  # 最小化 val_loss
        save_weights_only=True  # 只保存模型权重
    )
    lightning_setting = base_config['lightning']
    trainer = Trainer(**lightning_setting['lightning']['trainer'], logger=logger,
                      callbacks=[checkpoint_callback], devices=opt.gpus, max_epochs=1000)
    # Begin training or testing
    if not opt.test:
        trainer.fit(model, pl_dataset, ckpt_path=opt.resume)
    else:
        trainer.test(model, pl_dataset, ckpt_path=opt.resume)