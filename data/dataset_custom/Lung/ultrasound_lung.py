import os
import json

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class LungDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir (str): 数据根目录，包含 train 和 test 文件夹
            split (str): 数据集分割 ('train' 或 'test')
        """
        self.data_dir = data_dir
        self.split = split

        self.label_map = {'normal': 0, 'lesion': 1}

        self.json_file = os.path.join(self.data_dir, 'data_config.json')

        # 加载 JSON 文件
        with open(self.json_file, 'r') as f:
            self.labels = json.load(f)

        # 获取文件夹路径
        self.image_dir = os.path.join(self.data_dir, split)

        # 获取所有图像文件路径
        self.image_paths = []
        for image_name in os.listdir(self.image_dir):
            if image_name.endswith('.jpg'):
                self.image_paths.append(image_name)

        # 根据数据集分割选择不同的transform
        self.transform = self.get_transform()

    def get_transform(self):
        """返回适合当前数据集(split)的transform"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),  # 随机裁剪并调整大小
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 色彩抖动
                transforms.RandomRotation(30),  # 随机旋转
                transforms.ToTensor(),  # 转为Tensor
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),  # 调整图像大小
                transforms.CenterCrop(256),  # 中心裁剪
                transforms.ToTensor(),  # 转为Tensor
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图像文件名
        image_name = self.image_paths[idx]
        image_id = image_name.split('.')[0]  # 获取图像的 ID（去掉 .jpg 后缀）

        # 获取对应的标签
        image_label = self.labels[self.split].get(image_id, 'unknown')
        image_label = self.label_map[image_label]
        image_label = torch.tensor(image_label, dtype=torch.long)

        # 打开图像
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        # 如果有 transform 操作，应用它
        if self.transform:
            image = self.transform(image)

        return image, image_label

if __name__ == '__main__':
    lung_dataset = LungDataset(data_dir='/mnt/Dataset/Lung/process_data/cutting_data', split='train')
    img, label = lung_dataset.__getitem__(0)
    img = img.numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.savefig('lung.jpg')
    print(label)
    print(len(lung_dataset))