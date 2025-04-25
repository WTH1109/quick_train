import os
import cv2
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


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

        self.clip_length = 16
        self.clip_gap = 8
        self.clip_num = (self.target_length - self.clip_length) // self.clip_gap + 1

        self.key_frame_num_judge = 5

        self.score_list = []
        self.__init_label__()

    def __len__(self):
        return len(self.videos) * self.clip_num

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

    @staticmethod
    def pad_label_with_zeros(_label, target_length=250):
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
        _label = np.array(_label)
        current_length = len(_label)
        if current_length >= target_length:
            # 截断多余帧
            padded_label = _label[:target_length]
            return padded_label
        else:
            # 补零帧
            padded_label = np.pad(_label, (0, target_length - current_length), mode='constant')
            return padded_label

    def extract_frames_opencv(self, video_path, start_frame, end_frame, height=224, width=224):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

        # 检查 start_frame 和 end_frame 是否有效
        if start_frame < 0:
            raise ValueError("Start frame is out of range.")
        elif start_frame >= frame_count:
            frames = np.zeros((end_frame - start_frame, height, width, 3))
            return frames


        # 如果 end_frame 超出最大帧数，则调整 end_frame
        if end_frame >= frame_count:
            tmp_end_frame = frame_count - 1  # 最大帧数是 frame_count - 1
        else:
            tmp_end_frame = end_frame

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 设置开始读取的帧位置

        for _ in range(tmp_end_frame - start_frame):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.transform:
                frame = self.transform(frame)
            frame = np.array(frame)
            if not ret:
                # 如果读取失败，补 0（假设 frame 是 RGB 图像）
                frame = np.zeros((height, width, 3), dtype=np.uint8)
            frames.append(frame)

        # 如果 end_frame 超出范围，补 0
        if end_frame >= frame_count:
            for _ in range(end_frame - frame_count + 1):
                frames.append(np.zeros((height, width, 3), dtype=np.uint8))

        cap.release()
        frames = np.array(frames)

        return frames

    def __init_label__(self):
        for idx in range(self.__len__()):
            video_name = self.videos[idx // self.clip_num]
            _labels = self.data_info[self.mode][video_name]['key_frame_ssim']
            _labels = self.pad_label_with_zeros(_labels, self.target_length)
            clip_idx = idx % self.clip_num
            _labels = _labels[clip_idx * self.clip_gap: clip_idx * self.clip_gap + self.clip_length]
            _score = np.sum(_labels)
            if _score > self.key_frame_num_judge:
                _score = self.key_frame_num_judge
            _score /= self.key_frame_num_judge
            _score = 1 if _score > 0.5 else 0
            self.score_list.append(_score)
            self.weight_list = [score * 0.9 + 0.1 for score in self.score_list]


    def __getitem__(self, idx):
        video_name = self.videos[idx//self.clip_num]
        video_path = os.path.join(self.video_dir, self.mode, video_name)
        _labels = self.data_info[self.mode][video_name]['key_frame_ssim']
        _class = self.data_info[self.mode][video_name]['class']

        clip_idx = idx % self.clip_num
        _frames = self.extract_frames_opencv(video_path, start_frame=clip_idx * self.clip_gap,
                                   end_frame=clip_idx * self.clip_gap + self.clip_length, height=224, width=224)
        _labels = self.pad_label_with_zeros( _labels, self.target_length)
        _labels = _labels[clip_idx * self.clip_gap : clip_idx * self.clip_gap + self.clip_length]

        _score = np.sum(_labels)
        if _score > self.key_frame_num_judge:
            _score = self.key_frame_num_judge
        _score /= self.key_frame_num_judge
        _score = 1 if _score > 0.5 else 0

        # 转为 Tensor 并调整形状
        _frames = torch.tensor(_frames).permute(3, 0, 1, 2).float()  # (H, W, C) -> (C, T, H, W)
        _score = torch.tensor(_score).long()  # (T,)

        out_put = {
            'video': _frames,
            'score': _score,
        }

        return out_put

if __name__ == '__main__':
    print(torch.cuda.is_available())
    lung_dataset = FullVideoDataset(video_dir='/media/ps/data/home/wengtaohan/Dataset/video_dataset/rm_word', mode='test')
    print(len(lung_dataset))
    score_list = []
    score_cnt = 0  # train 304  test 81
    zero_cnt = 0   # train 3988  test 992
    for item in tqdm(range(37)):
        score_list = []
        score_num = 0
        for i in range(29):
            batch = lung_dataset.__getitem__(i+item*29)
            score_num += batch['score']
            score_list.append(batch['score'])
        score_cnt += score_num
        zero_cnt += 29 - score_num
        # print(score_list)
        # print(score_num)
    print(score_cnt)
    print(zero_cnt)
    print('done')

