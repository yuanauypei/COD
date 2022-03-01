import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2020-CVPR-SINet-0222/SINet_60.pth')
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-0222/')
parser.add_argument('--text_save', type=str,
                default='./Result/log/2020-CVPR-SINet-0222/')                
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()


for dataset in ['COD10K']:
    save_path = opt.test_save + dataset + '/'
    test_path = opt.test_save + dataset + '/log/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    f = open(test_path + 'log_50.txt', 'a+')
    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    test_loader = test_dataset(image_root='../../dataset/TestDataset/{}/img/'.format(dataset),
                               gt_root='../../dataset/TestDataset/{}/gt/'.format(dataset),
                               testsize=opt.testsize)
    mae_ave = 0
    for iteration in range(test_loader.size):
        # load data
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # inference
        _, cam = model(image)
        # reshape and squeeze
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        # misc.imsave(save_path+name, cam)
        # evaluate
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        mae_ave += mae
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, iteration+1,
                                                                           test_loader.size, mae)) 
        f.write('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, iteration+1,
                                                                           test_loader.size, mae) + "\n")
    mae_ave  = mae_ave / (iteration + 1)
    print('[Eval-Test] Dataset: {}, Total_nums: {}, MAE_AVE: {}'.format(dataset, iteration + 1, mae_ave))
    f.write('[Eval-Test] Dataset: {}, Total_nums: {}, MAE_AVE: {}'.format(dataset, iteration + 1, mae_ave))
    f.close()

print("\n[Congratulations! Testing Done]")
