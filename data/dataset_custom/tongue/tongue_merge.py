import os
import json
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import torchvision.transforms as transforms


class TongueDataset(Dataset):
    def __init__(self, data_dir, data_type='img', split='train', data_phase='tongue_seg', image_format='JPG',resize_size=512, sample_num=6):
        """
        Args:
            data_dir (str): 数据根目录，包含 train 和 test 文件夹
            split (str): 数据集分割 ('train' 或 'test')
        """
        self.data_dir = data_dir
        self.split = split
        self.resize_size = resize_size
        self.data_type = data_type

        self.label_map = {'no_Iga': 0, 'no_IgA': 0, 'IgA': 1, 'merge': 1}

        self.json_file = os.path.join(self.data_dir, 'dataset_info.json')

        self.image_format = image_format

        self.data_phase = data_phase
        self.img_path_idx_name = 'image_path'

        # 加载 JSON 文件
        with open(self.json_file, 'r') as f:
            self.dataset_info = json.load(f)

        self.data_info = self.dataset_info[self.split]

        # 根据数据集分割选择不同的transform
        self.transform = self.get_transform()

        if self.data_phase == 'tongue_region':
            self.sample_num = sample_num





    def get_transform(self):
        """返回适合当前数据集(split)的transform"""
        if self.split == 'train':
            if self.data_phase == 'tongue_seg':
                scale = random.uniform(0.6, 1.2)
                new_width = int(self.resize_size * scale)
                new_height = int(self.resize_size * scale)
                return transforms.Compose([
                    transforms.Resize(self.resize_size),  # 宽度为224，高度自动按纵横比缩放
                    transforms.CenterCrop(self.resize_size),  # 进行中心裁剪，确保图像是224x224
                    transforms.Resize((new_height, new_width), interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.Pad(
                        padding=(
                            (self.resize_size - new_width) // 2,  # 左
                            (self.resize_size - new_height) // 2,  # 上
                            (self.resize_size - new_width + 1) // 2,  # 右（处理奇数差值）
                            (self.resize_size - new_height + 1) // 2  # 下
                        ),
                        fill=0  # 填充黑色
                    ),
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 色彩抖动
                    transforms.RandomRotation(40),  # 随机旋转
                ])
            elif self.data_phase == 'tongue_sam':
                return transforms.Compose([
                    transforms.Resize(self.resize_size),  # 宽度为224，高度自动按纵横比缩放
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 色彩抖动
                    transforms.RandomRotation(40),  # 随机旋转
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(self.resize_size),  # 宽度为224，高度自动按纵横比缩放
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 色彩抖动
                    transforms.RandomRotation(40),  # 随机旋转
                ])
        else:
            return transforms.Compose([
                transforms.Resize(self.resize_size),  # 宽度为224，高度自动按纵横比缩放
                transforms.CenterCrop(self.resize_size),  # 进行中心裁剪，确保图像是224x224
            ])

    def __len__(self):
        if self.data_phase == 'tongue_region':
            return len(self.data_info) * self.sample_num
        else:
            return len(self.data_info)

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

    def __getitem__(self, idx):
        if self.data_phase == 'tongue_region':
            data_info = self.data_info[idx//self.sample_num]
        else:
            data_info = self.data_info[idx]

        if self.data_phase == 'tongue_region':
            image_path = data_info[self.img_path_idx_name]
            img_name = image_path.split('.')[0]
            sample_tmp_num = idx % self.sample_num
            image_path = img_name + '_%d'%sample_tmp_num + '.png'
            image_path = image_path.replace("tongue_IgA", "tongue_region_random_crop_IgA")
            image_path = image_path.replace("tongue_NOT_IgA", "tongue_region_random_crop_NOT_IgA")
        else:
            image_path = data_info[self.img_path_idx_name].replace('\\', '/')
        image_path = os.path.join(self.data_dir, image_path)

        patient_phase = data_info['label']
        image_label = self.label_map[patient_phase]
        if self.image_format == 'png':
            image_path = image_path.replace('jpg', 'png')
            image_path = image_path.replace('JPG', 'png')


        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)

        # new_height = int((self.resize_size / width) * height)
        # resized_image = cropped_img.resize((new_height, self.resize_size))

        resized_image = np.array(image).transpose(2, 0, 1) / 255.0

        resized_image = torch.tensor(resized_image).float()

        # 如果有 transform 操作，应用它
        if self.transform:
            resized_image = self.transform(resized_image)

        if self.data_type == 'multi':

            tabular_data_dic = {
                'gender': 1 if data_info['性别'] == '女' else 0,
                'age': data_info['年龄'] / 100.0,
                'BMI': (data_info['BMI'] - 10.0) / 30.0,
                'protein_24': data_info['24小时蛋白定量（g/24h）'] / 20.0,
                'protein': data_info['蛋白定量（g/l）'] / 8.0,
                'urine': data_info['尿量(L)'] / 10.0,
                'urine_protein': data_info['尿蛋白（mg/dl）【男0-17；女0-27】'] / 17.0
                if data_info['性别'] == '男' else data_info['尿蛋白（mg/dl）【男0-17；女0-27】'] / 27.0,
                'red_cell_urine': data_info['尿红细胞(/μL)'] / 6000.0,
                'blood_creatinine': (data_info['血肌酐（μmol/l）【女41-81；男57-111】'] - 57.0) / 54.0
                if data_info['性别'] == '男' else (data_info['血肌酐（μmol/l）【女41-81；男57-111】'] - 41.0) / 40.0,
                'urea_nitrogen': (data_info['尿素氮（mmol/l）【1.8-7.5】'] - 1.8) / 5.7,
                'albumin': (data_info['白蛋白(g/l)【35-50】'] - 35.0) / 15.0
            }
            tabular_tensor = self._dict_to_tensor(tabular_data_dic)
            return resized_image, tabular_tensor, image_label
        else:
            return resized_image, image_label

if __name__ == '__main__':
    lung_dataset = TongueDataset(data_dir='/media/ps/data-ssd/home-ssd/wengtaohan/Dataset/tongue/tongue_IgA',
                                 data_type='multi', split='train', data_phase='tongue_cutting', image_format='png')
    img, struct, label = lung_dataset.__getitem__(0)
    img = (img.numpy().transpose(1, 2, 0) * 255) .astype(np.uint8)
    plt.imshow(img)
    plt.savefig('img.jpg')

    pil_image = Image.fromarray(img)


    # 保存图像
    pil_image.save('output_image.jpg')

    print(label)
    print(len(lung_dataset))