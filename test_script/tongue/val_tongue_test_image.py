import os.path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from test_script.tongue.tongue_inference import TongueReference

os.chdir('../../')

Diagnostics = TongueReference(yaml_path='configs/base_config/tongue/ResNet_tongue_sam.yaml',
                            ckpt_path='logs/ResNet_tongue/D2025-03-18T19-53-25_ResNet34_tongue/checkpoints/epoch=100-val_loss=0.18.ckpt')

img_data_path = '/media/ps/data-ssd/home-ssd/wengtaohan/Dataset/tongue/val/val_tongue_cut'


xlsx_path = os.path.join(img_data_path, 'test.xlsx')

df = pd.read_excel(xlsx_path, sheet_name="Sheet1")

# 计算 BMI（体重(kg) / 身高(m)^2）
df["BMI"] = df["体重"] / (df["身高"] / 100) ** 2
# 按流水号提取为字典，格式: {流水号: {列名: 值, ...}}
data_all_dic = df.set_index('流水号').to_dict(orient="index")

for key, data_info in data_all_dic.items():
    tabular_data_dic = {
        'gender': 1 if data_info['性别'] == '女' else 0,
        'age': data_info['年龄'] / 100.0,
        'BMI': (data_info['BMI'] - 10.0) / 30.0,
        'protein_24': data_info['24小时蛋白定量'] / 20.0,
        'protein': data_info['蛋白定量（g/l）'] / 8.0,
        'urine': data_info['尿量'] / 10.0,
        'urine_protein': data_info['尿蛋白（mg/dl）【男0-17；女0-27】'] / 17.0
        if data_info['性别'] == '男' else data_info['尿蛋白（mg/dl）【男0-17；女0-27】'] / 27.0,
        'red_cell_urine': data_info['尿红细胞(/μL)（0-17）'] / 6000.0,
        'blood_creatinine': (data_info['血肌酐（μmol/l）【女41-81；男57-111】'] - 57.0) / 54.0
        if data_info['性别'] == '男' else (data_info['血肌酐（μmol/l）【女41-81；男57-111】'] - 41.0) / 40.0,
        'urea_nitrogen': (data_info['尿素氮（mmol/l）【1.8-7.5】'] - 1.8) / 5.7,
        'albumin': (data_info['白蛋白(g/l)【35-50】'] - 35.0) / 15.0
    }
    image_name = 'tonguedata_region_cut%sT.png' % str(key)
    image_path = os.path.join(img_data_path, image_name)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    result, prob = Diagnostics.reference(image)
    print(result)
    print(prob)
    print(key)
