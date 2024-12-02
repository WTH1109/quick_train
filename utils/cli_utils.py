import argparse
import os.path

import yaml
from torch.utils.hipify.hipify_python import str2bool


# Used for storing parameter parsing functions.
def main_opt_get(args=None):
    parser = main_get_parser()

    if args is not None:
        read_opt = parser.parse_args(args)
    else:
        read_opt = parser.parse_args()

    # Custom args setting
    if read_opt.name == '':
        read_opt.name = os.path.basename(read_opt.config).split('.')[-2]

    read_opt.config_name = os.path.basename(read_opt.config).split('.')[-2]

    return read_opt

def main_read_config(read_opt):
    with open(read_opt.config, 'r') as file:
        base_config = yaml.safe_load(file)
    with open(read_opt.lightning_config, 'r') as file:
        lightning_config = yaml.safe_load(file)
        base_config['lightning'] = lightning_config
    if read_opt.dataset_config is not None:
        with open(read_opt.dataset_config, 'r') as file:
            data_config = yaml.safe_load(file)
            base_config['data'] = data_config
    return base_config


def main_get_parser():

    parser = argparse.ArgumentParser(description="main training parser")

    parser.add_argument('-n', '--name', type=str, default='', help='save model name')
    parser.add_argument('-t', '--test', type=str2bool, default=False, help='choose testing the model.')
    parser.add_argument('-g', '--gpus', type=str, default='0,', help='select the gpus to use.')
    parser.add_argument('-r', '--resume', type=str, default=None, help='last.ckpt path')

    parser.add_argument('--debug', type=str, default=False,
                        help='quickly run one epoch to validate the correctness of the code.')

    # Set configuration parameters.
    parser.add_argument('-c',
                        '--config',
                        required=True,
                        type=str,
                        metavar="base_config.yaml",
                        help='path to base default_configs.')

    parser.add_argument('-lc',
                        '--lightning_config',
                        type=str,
                        default="default_configs/lightning_config/default_lightning.yaml",
                        metavar="lightning_config.yaml",
                        help='path to lightning default_configs. Including training Strategy, batch size....')

    parser.add_argument('-dc',
                        '--dataset_config',
                        type=str,
                        default=None,
                        metavar="dataset_config.yaml",
                        help='Paths to dataset default_configs.'
                        'if you do not set this parameter, it will default to using the dataset in the config.')

    return parser

if __name__ == '__main__':
    opt = main_opt_get(['-c', '../default_configs/base_config/stable_diffusion/sd21_ge_control.yaml'])
    main_read_config(opt)