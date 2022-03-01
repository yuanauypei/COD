import os
import torch
import argparse
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import get_loader
from Src.utils.trainer import trainer, adjust_lr
from apex import amp
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--save_model', type=str, default='./Snapshot/2020-CVPR-SINet_BofRGB_50/')
    parser.add_argument('--train_img_dir', type=str, default="/home/hebichang/CamDetector/dataset/TrainDataset/img/")
    parser.add_argument('--train_gt_dir', type=str, default='/home/hebichang/CamDetector/dataset/TrainDataset/gt/')
    opt = parser.parse_args()
    
    # torch.cuda.set_device(opt.gpu)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.set_device(device)

    # TIPS: you also can use deeper network for better performance like channel=64
    model_SINet = SINet_ResNet50(channel=32)
    print('-' * 30, model_SINet, '-' * 30)

    

    optimizer = torch.optim.Adam(model_SINet.parameters(), opt.lr)
    LogitsBCE = torch.nn.BCEWithLogitsLoss()
    
    net, optimizer = amp.initialize(model_SINet.to(device), optimizer, opt_level='O1')     # NOTES: Ox not 0x
    
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    
    train_loader = get_loader(opt.train_img_dir, opt.train_gt_dir, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, num_workers=12)
    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.lr,
                                                              opt.batchsize, opt.save_model, total_step), '-' * 30)

    for epoch_iter in range(1, opt.epoch):
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
        trainer(train_loader=train_loader, model=model_SINet,
                optimizer=optimizer, epoch=epoch_iter,
                opt=opt, loss_func=LogitsBCE, total_step=total_step)
