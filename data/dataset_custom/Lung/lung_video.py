import os
import cv2
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms




class FullVideoDataset(Dataset):
    def __init__(self, video_dir, mode='train', target_length=240, top_k=5):
        """
        Args:
            video_dir (str): 视频目录。
            mode (str): 'train' 或 'test'。
        """
        self.video_dir = video_dir
        self.json_path = os.path.join(self.video_dir, 'data_info.json')
        self.mode = mode
        self.top_k = top_k
        self.target_length = target_length

        # 加载标注文件
        with open(self.json_path, 'r') as f:
            self.data_info = json.load(f)

        # 获取视频文件列表
        self.videos = list(self.data_info[self.mode].keys())
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.videos)

    def get_transform(self):
        if self.mode == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并调整大小
                # transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 色彩抖动
                # transforms.RandomRotation(30),  # 随机旋转
                # transforms.ToTensor(),  # 转为Tensor
            ])
        else:
            return transforms.Compose([
                transforms.Resize(224),  # 调整图像大小
                transforms.CenterCrop(224),  # 中心裁剪
                # transforms.ToTensor(),  # 转为Tensor
            ])

    @staticmethod
    def pad_video_with_zeros(_frames, _label, target_length=250):
        """
        将视频帧补充到固定长度，采用补零方式。
        Args:
            _frames (list or np.ndarray): 视频帧列表，形状为 (T, H, W, C)。
            _label: 视频关键帧label，输入为list
            target_length (int): 目标帧数，默认为 250。
        Returns:
            np.ndarray: 补充后的帧，形状为 (target_length, H, W, C)。
            :param target_length:
            :param _frames:
            :param _label:
        """
        d, h, w, c = _frames.shape
        _label = np.array(_label)
        current_length = len(_frames)
        if current_length >= target_length:
            # 截断多余帧
            padded_label = _label[:target_length]
            return _frames[:target_length], padded_label
        else:
            # 补零帧
            padded_label = np.pad(_label, (0, target_length - current_length), mode='constant')
            zero_frames = np.zeros((target_length - current_length, h, w, c), dtype=np.float32)
            return np.concatenate((_frames, zero_frames), axis=0), padded_label

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        video_path = os.path.join(self.video_dir, self.mode, video_name)
        _labels = self.data_info[self.mode][video_name]['key_frame_ssim']
        _class = self.data_info[self.mode][video_name]['class']

        # 加载视频帧
        cap = cv2.VideoCapture(video_path)
        _frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frame = np.array(frame)
            _frames.append(frame)
        cap.release()

        _frames = np.stack(_frames, axis=0)
        _frames, _labels = self.pad_video_with_zeros(_frames, _labels, self.target_length)

        if _class == 'lesion':
            # 找到最大的5个label作为要预测的内容
            top_k_indices = np.argsort(_labels)[-self.top_k:][::-1].copy() # (num_class, )
            one_hot_label = np.zeros_like(_labels, dtype=np.float32)
            one_hot_label[top_k_indices] = 1.0
            one_hot_label = np.append(one_hot_label, 1.0)
        else:
            top_k_indices = [0 for _ in range(self.top_k)]
            one_hot_label = np.zeros_like(_labels, dtype=np.float32)
            one_hot_label = np.append(one_hot_label, 0)

        # 转为 Tensor 并调整形状
        _frames = torch.tensor(_frames).permute(3, 0, 1, 2).float()  # (H, W, C) -> (C, T, H, W)
        _labels = torch.tensor(_labels).float()  # (T,)
        one_hot_label = torch.tensor(one_hot_label).float()  # (T,)

        out_put = {
            'video_frame': _frames,
            'one_hot_label': one_hot_label,
            'ori_label': _labels,
            'top_k_indices': top_k_indices,
        }

        return out_put

if __name__ == '__main__':
    print(torch.cuda.is_available())
    lung_dataset = FullVideoDataset(video_dir='/mnt/Dataset/video_dataset/rm_word', mode='test')
    for i in range(36):
        batch = lung_dataset.__getitem__(i)
        print('step')
    print('done')

