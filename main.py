import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from utils.cli import main_opt_get, main_read_config
from utils.instance_from_config import instantiate_from_config

if __name__ == '__main__':

    opt = main_opt_get()
    base_config = main_read_config(opt)

    pl_dataset = instantiate_from_config(base_config['data'])
    model = instantiate_from_config(base_config['model'])

    lightning_setting = base_config['lightning']

    logger = TensorBoardLogger("tb_logs", name=opt.name)
    trainer = Trainer(**lightning_setting['train'], logger=logger)

    if opt.test:
        Trainer.test(model, pl_dataset)
    else:
        Trainer.fit(model, pl_dataset)