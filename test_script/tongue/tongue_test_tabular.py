import os

import torch
from matplotlib import pyplot as plt

from utils.cli_utils import main_opt_get, main_read_config
from utils.config_utils import instantiate_from_config

if __name__ == '__main__':
    os.chdir('../../')
    print(os.getcwd())
    # Parse Command Line Arguments.
    opt = main_opt_get(['-c', '/media/ps/data/home/wengtaohan/Code/quick_train/configs/base_config/tongue/ResNet_tongue_tabular.yaml'])
    base_config = main_read_config(opt)
    # base_config['model']['params']['pretrained'] = False
    # Instantiate dataloader and model.
    pl_dataset = instantiate_from_config(base_config['data'])
    model = instantiate_from_config(base_config['model'])

    model_dict = torch.load('/media/ps/data/home/wengtaohan/Code/quick_train/logs/ResNet_tongue_tabular/D2025-03-17T17-05-55_ResNet_tongue_tabular.yaml/checkpoints/epoch=23-val_loss=0.60.ckpt'
                            ,weights_only=True)["state_dict"]
    missing_key, unexpect_key = model.load_state_dict(model_dict, strict=False)
    print('missing key:')
    print('******************************************')
    print(missing_key)
    print('unexpect key:')
    print('******************************************')
    print('******************************************')
    print(unexpect_key)

    test_data_set = instantiate_from_config(base_config['data']['params']['test'])

    dataset_len = len(test_data_set)
    save_dir = '.images/'
    os.makedirs(save_dir, exist_ok=True)
    for i in range(dataset_len):
        img, label = test_data_set.__getitem__(i)
        ori_img = img.detach().cpu().numpy()

        img = torch.unsqueeze(img, 0)

        model.eval()

        predict = model(img)
        predict = torch.max(predict, 1)[1][0].detach().cpu().numpy()
        label = label.detach().cpu().numpy()

        label_list = ['normal', 'lesion']

        # print(predict)
        # print(label)

        plt.imshow(ori_img.transpose(1, 2, 0))
        plt.title('label: %s predict:%s'%(label_list[label],label_list[predict]))
        plt.savefig(os.path.join(save_dir, '%03d.jpg'%i))