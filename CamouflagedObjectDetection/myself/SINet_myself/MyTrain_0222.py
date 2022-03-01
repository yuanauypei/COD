import os
import torch
from torch import distributed, optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import CamObjDataset
from Src.utils.trainer_multigpu import trainer, adjust_lr

from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--epoch', type=int, default=200,
                        help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=24,
                        help='training batch size (Note: ~500MB per img in GPU)')#36
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='every N epochs decay lr')
    parser.add_argument('--gpu', type=str, default='0, 1, 2, 3',
                        help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/2020-CVPR-SINet-0222/')
    parser.add_argument('--train_img_dir', type=str, default="/home/hebichang/CamDetector/dataset/TrainDataset/img/")
    parser.add_argument('--train_gt_dir', type=str, default='/home/hebichang/CamDetector/dataset/TrainDataset/gt/')
    opt = parser.parse_args()
    return opt

def main():
    opt = parse()
    
    # 设置当前进程的device GPU的通信方式为NCCL
    torch.cuda.set_device(opt.local_rank)
    distributed.init_process_group('nccl', init_method='env://')

    # 生成Dataset和Sampler
    dataset = CamObjDataset(opt.train_img_dir, opt.train_gt_dir, opt.trainsize)
    train_sampler = DistributedSampler(dataset)
    train_loader = data.DataLoader(dataset=dataset,
                                  batch_size=opt.batchsize,
                                  # shuffle=opt.shuffle,
                                  num_workers=opt.num_workers,
                                  pin_memory=opt.pin_memory,
                                  sampler=train_sampler)
    
    device = torch.device('cuda:{}'.format(opt.local_rank))

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(SINet_ResNet50(channel=32)).to(device)
    net = SINet_ResNet50(channel=32).to(device)
    optimizer = torch.optim.Adam(net.parameters(), opt.lr)

    net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
    net = DistributedDataParallel(net, delay_allreduce=True)
     
    LogitsBCE = torch.nn.BCEWithLogitsLoss().to(device)

    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.lr,
                                                              opt.batchsize, opt.save_model, total_step), '-' * 30)
    
    for epoch_iter in range(1, opt.epoch):
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
        trainer(train_loader=train_loader, model=net,
                optimizer=optimizer, epoch=epoch_iter,
                opt=opt, loss_func=LogitsBCE, total_step=total_step, device=device)
    

  
if __name__ == "__main__":
    main()

    
