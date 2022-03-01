# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import sys
import cv2
from tqdm import tqdm

# pip install pysodmetrics
sys.path.append("/home/zhuyuan/zy_all/CamouflagedObjectDetection/SINet-master/EvaluationTool/")
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

FM = Fmeasure()
WFM = WeightedFmeasure()
SM = Smeasure()
EM = Emeasure()
MAE = MAE()

# data_root = "./test_data"
mask_root = "/media/disk1/zhuyuan/datasets/CamouflagedObjectDetection/TestDataset/CAMO_TestingDataset/GT/"
pred_root = "/home/zhuyuan/zy_all/CamouflagedObjectDetection/del/SINet-master/Result/2020-CVPR-SINet-New/CAMO_TestingDataset/"
mask_name_list = sorted(os.listdir(mask_root))
for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
    mask_path = os.path.join(mask_root, mask_name)
    pred_path = os.path.join(pred_root, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    FM.step(pred=pred, gt=mask)
    WFM.step(pred=pred, gt=mask)
    SM.step(pred=pred, gt=mask)
    EM.step(pred=pred, gt=mask)
    MAE.step(pred=pred, gt=mask)

fm = FM.get_results()["fm"]
wfm = WFM.get_results()["wfm"]
sm = SM.get_results()["sm"]
em = EM.get_results()["em"]
mae = MAE.get_results()["mae"]

results = {
    "Smeasure": sm,
    "wFmeasure": wfm,
    "MAE": mae,
    "adpEm": em["adp"],
    "meanEm": em["curve"].mean(),
    "maxEm": em["curve"].max(),
    "adpFm": fm["adp"],
    "meanFm": fm["curve"].mean(),
    "maxFm": fm["curve"].max(),
}

print(results)
# 'Smeasure': 0.9029763868504661,
# 'wFmeasure': 0.5579812753638986,
# 'MAE': 0.03705558476661653,
# 'adpEm': 0.9408760066970631,
# 'meanEm': 0.9566258293508715,
# 'maxEm': 0.966954482892271,
# 'adpFm': 0.5816750824038355,
# 'meanFm': 0.577051059518767,
# 'maxFm': 0.5886784581120638

# version 1.2.3
# 'Smeasure': 0.9029763868504661,
# 'wFmeasure': 0.5579812753638986,
# 'MAE': 0.03705558476661653,
# 'adpEm': 0.9408760066970631,
# 'meanEm': 0.9566258293508715,
# 'maxEm': 0.966954482892271,
# 'adpFm': 0.5816750824038355,
# 'meanFm': 0.577051059518767,
# 'maxFm': 0.5886784581120638
