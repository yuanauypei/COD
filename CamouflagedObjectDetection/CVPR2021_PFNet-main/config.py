"""
 @Time    : 2021/7/6 09:46
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2021_PFNet
 @File    : config.py
 @Function: Configuration
 
"""
'''
import os

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

datasets_root = '../data/NEW'

cod_training_root = os.path.join(datasets_root, 'train')

chameleon_path = os.path.join(datasets_root, 'test/CHAMELEON')
camo_path = os.path.join(datasets_root, 'test/CAMO')
cod10k_path = os.path.join(datasets_root, 'test/COD10K')
nc4k_path = os.path.join(datasets_root, 'test/NC4K')
'''

backbone_path = './backbone/resnet/resnet50-19c8e357.pth'

cod_training_root = "/media/disk1/zhuyuan/datasets/CamouflagedObjectDetection/COD10K_CAMO_CombinedTrainingDataset/"

chameleon_path = "/media/disk1/zhuyuan/datasets/CamouflagedObjectDetection/TestDataset/CHAMELEON_TestingDataset/"
camo_path = "/media/disk1/zhuyuan/datasets/CamouflagedObjectDetection/TestDataset/CAMO_TestingDataset/"
cod10k_path = "/media/disk1/zhuyuan/datasets/CamouflagedObjectDetection/TestDataset/COD10K_TestingDataset/"
# nc4k_path = os.path.join(datasets_root, 'test/NC4K')

