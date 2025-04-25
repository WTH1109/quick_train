import json
import os.path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from test_script.tongue.tongue_inference import TongueReference

os.chdir('../../')

Diagnostics = TongueReference(yaml_path='configs/base_config/tongue/ResNet_tongue_region_multi.yaml',
                            ckpt_path='logs/ResNet_tongue_region_multi/D2025-03-22T13-26-47_tongue_region_multi/checkpoints/epoch=10-val_loss=0.48.ckpt')

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
    img_name = image_path.split('.')[0]
    image_path = img_name + '_%d' % 0 + '.png'
    image_path = image_path.replace("tongue_cutting_IgA", "tongue_region_random_crop_IgA")
    image_path = image_path.replace("tongue_cutting_NOT_IgA", "tongue_region_random_crop_NOT_IgA")
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)

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

    result, prob = Diagnostics.reference_multi(image, tabular_data_dic)
    gt = data_info['patient_phase']
    if result == gt:
        acc_cnt += 1
    total_cnt += 1

acc = acc_cnt / total_cnt * 100.0
print(acc)
print(acc_cnt)
print(total_cnt)
