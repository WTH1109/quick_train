import torch

from utils.cli_utils import main_opt_get, main_read_config
from utils.config_utils import instantiate_from_config
from torchvision import transforms

class TongueReference:
    def __init__(self, yaml_path, ckpt_path):
        """

        :param yaml_path: 模型配置文件
        :param ckpt_path: 模型保存文件
        """
        opt = main_opt_get(['-c', yaml_path])
        base_config = main_read_config(opt)
        # if base_config['model']['params']['pretrained']
        # base_config['model']['params']['pretrained'] = False

        # Instantiate dataloader and model.
        self.model = instantiate_from_config(base_config['model'])
        self.model.eval()

        self.model.load_state_dict(torch.load(ckpt_path, weights_only=True)["state_dict"], strict=False)

    def _dict_to_tensor(self, data_dict):
        """将字典转换为张量（不应用归一化）"""
        return torch.tensor([
            data_dict['gender'],
            data_dict['age'],
            data_dict['BMI'],
            data_dict['protein_24'],
            data_dict['protein'],
            data_dict['urine'],
            data_dict['urine_protein'],
            data_dict['red_cell_urine'],
            data_dict['blood_creatinine'],
            data_dict['urea_nitrogen'],
            data_dict['albumin']
        ], dtype=torch.float32)

    def reference(self, image):
        """

        :param image: PIL image
        :return: label
        """
        img_transform = transforms.Compose([
            transforms.Resize(512),  # 调整图像大小
            transforms.CenterCrop(512),  # 中心裁剪
            transforms.ToTensor(),  # 转为Tensor
        ])
        image = img_transform(image)
        img = torch.unsqueeze(image, 0)

        predict = self.model(img)
        prob = torch.softmax(predict, dim=1)
        predict = torch.max(predict, 1)[1][0].detach().cpu().numpy()
        label_list = ['no_IgA', 'IgA']

        return label_list[predict], prob

    def reference_multi(self, image, tabular_data_dic):
        img_transform = transforms.Compose([
            transforms.Resize(256),  # 调整图像大小
            transforms.CenterCrop(256),  # 中心裁剪
            transforms.ToTensor(),  # 转为Tensor
        ])
        image = img_transform(image)
        img = torch.unsqueeze(image, 0)

        tabular_tensor = self._dict_to_tensor(tabular_data_dic)
        tabular_tensor = tabular_tensor.unsqueeze(0)

        predict = self.model(img, tabular_tensor)
        prob = torch.softmax(predict, dim=1)
        predict = torch.max(predict, 1)[1][0].detach().cpu().numpy()
        label_list = ['no_IgA', 'IgA']

        return label_list[predict], prob
