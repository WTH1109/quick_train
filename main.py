from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from utils.cli_utils import main_opt_get, main_read_config
from utils.config_utils import instantiate_from_config
from utils.file_utils import MainFileManage

if __name__ == '__main__':

    opt = main_opt_get(['-c', 'default_configs/base_config/stable_diffusion/sd21_ge_control.yaml'])
    base_config = main_read_config(opt)

    # pl_dataset = instantiate_from_config(base_config['data'])
    # model = instantiate_from_config(base_config['model'])

    lightning_setting = base_config['lightning']
    log_setting = MainFileManage(opt.config_name, opt.name, base_config)

    logger = TensorBoardLogger(log_setting.save_tensorboard_log_path, name=opt.name)
    trainer = Trainer(**lightning_setting['train'], logger=logger)

    # if opt.test:
    #     Trainer.test(model, pl_dataset)
    # else:
    #     Trainer.fit(model, pl_dataset)