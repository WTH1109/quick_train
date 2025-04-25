import torch

from utils.cli_utils import main_opt_get, main_read_config
from utils.config_utils import instantiate_from_config
from torchvision import transforms

class LungReference:
    def __init__(self, yaml_path, ckpt_path):
        """

        :param yaml_path: 模型配置文件
        :param ckpt_path: 模型保存文件
        """
        opt = main_opt_get(['-c', yaml_path])
        base_config = main_read_config(opt)
        base_config['model']['params']['pretrained'] = False

        # Instantiate dataloader and model.
        self.model = instantiate_from_config(base_config['model'])
        self.model.eval()

        self.model.load_state_dict(torch.load(ckpt_path, weights_only=True)["state_dict"], strict=False)

    def reference(self, image):
        """

        :param image: PIL image
        :return: label
        """
        img_transform = transforms.Compose([
            transforms.Resize(256),  # 调整图像大小
            transforms.CenterCrop(256),  # 中心裁剪
            transforms.ToTensor(),  # 转为Tensor
        ])
        image = img_transform(image)
        img = torch.unsqueeze(image, 0)

        predict = self.model(img)
        predict = torch.max(predict, 1)[1][0].detach().cpu().numpy()
        label_list = ['normal', 'lesion']

        return label_list[predict]
