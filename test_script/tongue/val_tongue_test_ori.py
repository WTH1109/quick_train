import json
import os.path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from test_script.tongue.tongue_inference import TongueReference

os.chdir('../../')

Diagnostics = TongueReference(yaml_path='configs/base_config/tongue/ResNet_tongue_sam.yaml',
                            ckpt_path='logs/ResNet_tongue/D2025-03-18T19-53-25_ResNet34_tongue/checkpoints/epoch=420-val_loss=0.18.ckpt')

img_data_path = '/media/ps/data-ssd/home-ssd/wengtaohan/Dataset/tongue/val/val_tongue_cut'

json_file = '/media/ps/data-ssd/home-ssd/wengtaohan/Dataset/tongue/tongue_IgA/train_test_data_info.json'

# 加载 JSON 文件
with open(json_file, 'r') as f:
    dataset_info = json.load(f)

data_all_dic = dataset_info['test']

# xlsx_path = os.path.join(img_data_path, 'test.xlsx')

acc_cnt = 0
total_cnt = 0

for data_info in tqdm(data_all_dic):
    image_name = data_info['cutting_tongue_path'].replace('JPG', 'png')
    image_path = os.path.join('/media/ps/data-ssd/home-ssd/wengtaohan/Dataset/tongue/tongue_IgA', image_name)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

    result, prob = Diagnostics.reference(image)
    gt = data_info['patient_phase']
    if result == gt:
        acc_cnt += 1
    total_cnt += 1

acc = acc_cnt / total_cnt * 100.0
print(acc)
print(acc_cnt)
print(total_cnt)
