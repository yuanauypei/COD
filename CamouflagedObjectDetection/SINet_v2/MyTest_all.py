import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2020-CVPR-SINet2/')  #
# parser.add_argument('--test_save', type=str,
#                     default='./Result/2020-CVPR-SINet-New/')
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-del/')
parser.add_argument('--txt_save', type=str,
                    default='./Result/2020-CVPR-SINet-del/maeres.txt')
opt = parser.parse_args()

models = os.listdir(opt.model_path)
model = SINet_ResNet50().cuda()


for dataset in ['CAMO_TestingDataset']:
    best_avg_mae = 1000
    best_model = None
    for model_it in models:
        now_model = opt.model_path + '/' + model_it
        # pdb.set_trace()
        model.load_state_dict(torch.load(now_model))
        model.eval()
        save_path = opt.test_save + dataset + model_it + '/'
        os.makedirs(save_path, exist_ok=True)
        # NOTES:
        #  if you plan to inference on your customized dataset without grouth-truth,
        #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
        #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
        #  the grouth-truth map is unnecessary actually.
        test_loader = test_dataset(image_root='/media/disk1/zhuyuan/datasets/CamouflagedObjectDetection/TestDataset/{}/Image/'.format(dataset),
                                gt_root='/media/disk1/zhuyuan/datasets/CamouflagedObjectDetection/TestDataset/{}/GT/'.format(dataset),
                                testsize=opt.testsize)
        img_count = 0
        maeall = 0
        for iteration in range(test_loader.size):
            # load data
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            # inference
            # pdb.set_trace()
            _, cam = model(image)
            # reshape and squeeze
            cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
            cam = cam.sigmoid().data.cpu().numpy().squeeze()
            # normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            misc.imsave(save_path+name, cam)
            # evaluate
            mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
            maeall = maeall + mae
            content = '[Eval-Test] Dataset: {}, Image: {} ({}/{}), model: {}, MAE: {}\n'.format(dataset, name, img_count,
                                                                            test_loader.size,model_it,mae)
            # coarse score
            with open(opt.txt_save, 'a') as f:
                f.write(content)

            print(content)
            img_count += 1
        print("数量：", img_count)
        avg_mae = maeall/img_count
        content2 = '[Eval-Test] Dataset: {}, model:{}, avg_mae: {}\n'.format(dataset,model_it,avg_mae)
        if avg_mae<best_avg_mae:
            best_avg_mae = avg_mae
            best_model = model_it
        content3 = '[Eval-Test] Dataset: {}, best_avg_mae: {}, best_model: {}\n'.format(dataset, best_avg_mae, best_model)
        with open(opt.txt_save, 'a') as f2:
            f2.write(content2)
            f2.write(content3)
        # pdb.set_trace()
        print(content2)
        print(content3)
        pdb.set_trace()
        
   
        
f.close()
f2.close()
print("\n[Congratulations! Testing Done]")
