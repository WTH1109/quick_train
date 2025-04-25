import os
from collections import Counter

import cv2
from PIL import Image

from test_script.lung.lung_reference import LungReference

video_path = '/media/ps/data-ssd/home-ssd/wengtaohan/Dataset/Lung/process_data/video_frame_data/test_video/video_data/'



def judge_video(_video_path, detector, num_frames=15):
    cap = cv2.VideoCapture(_video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return []

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算间隔
    frame_interval = total_frames // num_frames

    detect_result_list = []
    for i in range(num_frames):
        # 跳到指定的帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)

        # 读取该帧
        ret, frame = cap.read()
        if ret:
            frame = Image.fromarray(frame)
            detect_result = detector.reference(frame)
            detect_result_list.append(detect_result)
        else:
            print(f"Error: Failed to read frame {i}")
            break

    # 释放视频对象
    cap.release()
    return detect_result_list

def count_elements(lst):
    # 使用 Counter 来计算每个元素的频率
    counter = Counter(lst)
    return counter


def integration_diagnosis(_video_path, detector, _num_frames=15):
    result_list = judge_video(_video_path, detector, num_frames=_num_frames)
    _integration_result = count_elements(result_list)
    lesion_rate = _integration_result['lesion'] / _num_frames
    # print('lesion cnt:%d  normal cnt:%d'%(_integration_result['lesion'], _integration_result['normal']))
    # print(lesion_rate)
    if lesion_rate > 0.3:
        return 'lesion'
    else:
        return 'normal'

os.chdir('../../')

Diagnostics = LungReference(yaml_path='configs/base_config/Lung/ResNet_video_frame.yaml',
                            ckpt_path='logs/ResNet_video_frame/D2025-01-19T10-47-51_ResNet_video_frame/checkpoints/epoch=12-val_loss=0.37.ckpt')
# Diagnostics = LungReference(yaml_path='configs/base_config/Lung/ResNet_Total.yaml',
#                             ckpt_path='logs/ResNet_Total/D2024-12-06T03-09-50_ResNet_Total/checkpoints/epoch=68-val_loss=0.06.ckpt')

num_frames = 25
phase = 'lesion'
video_list = os.listdir(os.path.join(video_path, phase))
for video_name in video_list:
    tmp_video_path = os.path.join(video_path, phase, video_name)
    integration_result = integration_diagnosis(tmp_video_path, Diagnostics, num_frames)
    print('label:lesion  pred:%s'%integration_result)


phase = 'normal'
video_list = os.listdir(os.path.join(video_path, phase))
for video_name in video_list:
    tmp_video_path = os.path.join(video_path, phase, video_name)
    integration_result = integration_diagnosis(tmp_video_path, Diagnostics, num_frames)
    print('label:normal  pred:%s'%integration_result)