#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os

import numpy as np
import torch

from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,StepLR

import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset.datasets import WLFWDatasets
import models.pfld
import models.ghostnetv3
import models.mobilenetv4
import models.mobilenetv4_gcon
#from models.pfld import PFLDInference, AuxiliaryNet
#from models.ghostnetv3 import GhostStarNet,AuxiliaryNet
from pfld.loss import PFLDLoss
from pfld.utils import AverageMeter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def train(train_loader, pfld_backbone, auxiliarynet, criterion, optimizer
          , epoch):
    losses = AverageMeter()

    weighted_loss, loss = None, None
    for img, landmark_gt, attribute_gt, euler_angle_gt in train_loader:
        img = img.to(device)
        attribute_gt = attribute_gt.to(device)
        landmark_gt = landmark_gt.to(device)
        euler_angle_gt = euler_angle_gt.to(device)
        pfld_backbone = pfld_backbone.to(device)
        auxiliarynet = auxiliarynet.to(device)
        features, landmarks = pfld_backbone(img)
        angle = auxiliarynet(features)
        weighted_loss, loss = criterion(attribute_gt, landmark_gt,
                                        euler_angle_gt, angle, landmarks,
                                        args.train_batchsize)
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        

        losses.update(loss.item())
    return weighted_loss, loss


def validate(wlfw_val_dataloader, pfld_backbone, auxiliarynet, criterion):
    pfld_backbone.eval()
    auxiliarynet.eval()
    losses = []
    with torch.no_grad():
        for img, landmark_gt, attribute_gt, euler_angle_gt in wlfw_val_dataloader:
            img = img.to(device)
            attribute_gt = attribute_gt.to(device)
            landmark_gt = landmark_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            auxiliarynet = auxiliarynet.to(device)
            _, landmark = pfld_backbone(img)
            loss = torch.mean(torch.sum((landmark_gt - landmark)**2, axis=1))
            losses.append(loss.cpu().numpy())
    print("===> Evaluate:")
    print('Eval set: Average loss: {:.4f} '.format(np.mean(losses)))
    return np.mean(losses)


def main(args):
    # Step 1: parse args config
    logging.basicConfig(
        format=
        '[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, criterion, optimizer, scheduler
    ## chooose model

    if args.model == 'pfld':
        pfld_backbone = models.pfld.PFLDInference().to(device)
        auxiliarynet = models.pfld.AuxiliaryNet().to(device)
    elif args.model == 'ghostnetv3':
        pfld_backbone = models.ghostnetv3.GhostStarNet().to(device)
        auxiliarynet = models.ghostnetv3.AuxiliaryNet().to(device)
    elif args.model =='mobilenetv4_small_OriHeater':
        '''
        该模型并非原始模型，推理速度，精度整体符合要求，略低于PFLD
        该模型原本输入224，下载输入112，导致stage对应的size不太一样
        
        '''
        pfld_backbone = models.mobilenetv4.mobilenetv4_conv_small().to(device)
        auxiliarynet = models.mobilenetv4.AuxiliaryNet().to(device)
    elif args.model =='mobilenetv4_small_bockbone_fixed':
        '''
        该模型仅仅改变backbone的参数，调整了以下stage对应的pix size,推理速度5ms，比原版mobilenetv4_conv_small慢了一秒
        准确率8.9，与pfld少了两个点
        '''
        pfld_backbone = models.mobilenetv4.mobilenetv4_conv_small_bockbone_fixed().to(device)
        auxiliarynet = models.mobilenetv4.AuxiliaryNet().to(device)
    elif args.model =='mobilenetv4_ori':
        '''
        mobilenetv4_conv_small_ori
        未修改baobone，仅仅删除了一个下采样层，推理速度，后期模型与一开始迭代的模型翻倍（偶发情况），精度比上一个高，采用原始的mobilenetV4检测头
        loss能达到0.2011,推理速度5.6ms
        mobilenetv4_conv_small_ori1,loss差不多，但是推理速度慢很多，8.9ms
        '''
        pfld_backbone = models.mobilenetv4_gcon.mobilenetv4_conv_small_ori().to(device)
        auxiliarynet = models.mobilenetv4_gcon.AuxiliaryNet_Ori().to(device)
    elif args.model =='mobilenetv4_gcon':
        '''
        原始模型，采用gcon检测头，

        '''
        pfld_backbone = models.mobilenetv4_gcon.mobilenetv4_conv_small_gcon().to(device)
        auxiliarynet = models.mobilenetv4_gcon.AuxiliaryNet_Ori().to(device)



    else:
        ##model not support
        raise Exception("模型不支持")
        
    #pfld_backbone = PFLDInference().to(device)
    # pfld_backbone = GhostStarNet().to(device)
    # auxiliarynet = AuxiliaryNet().to(device)
    criterion = PFLDLoss()
    ##默认使用adam优化器，已修改
    optimizer = torch.optim.Adam([{
        'params': pfld_backbone.parameters()
    }, {
        'params': auxiliarynet.parameters()
    }],
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    
    # 定义学习率调度器
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-10)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', patience=args.lr_patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=args.lr_patience, min_lr=1e-30)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.8, verbose=True)
    if args.resume:
        checkpoint = torch.load(args.resume)
        auxiliarynet.load_state_dict(checkpoint["auxiliarynet"])
        pfld_backbone.load_state_dict(checkpoint["pfld_backbone"])
        args.start_epoch = checkpoint["epoch"]

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    wlfwdataset = WLFWDatasets(args.dataroot, transform)
    dataloader = DataLoader(wlfwdataset,
                            batch_size=args.train_batchsize,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=False)

    wlfw_val_dataset = WLFWDatasets(args.val_dataroot, transform)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset,
                                     batch_size=args.val_batchsize,
                                     shuffle=False,
                                     num_workers=args.workers)

    # step 4: run
    writer = SummaryWriter(args.tensorboard)
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, pfld_backbone,
                                                auxiliarynet, criterion,
                                                optimizer,
                                                 epoch)
        filename = os.path.join(str(args.snapshot),
                                "checkpoint_epoch_" + str(epoch) + '.pth.tar')

        save_checkpoint(
            {
                'epoch': epoch,
                'pfld_backbone': pfld_backbone.state_dict(),
                # 'auxiliarynet': auxiliarynet.state_dict()
            }, filename)

        val_loss = validate(wlfw_val_dataloader, pfld_backbone, auxiliarynet,
                            criterion)
        #根据损失设置学习率
        scheduler.step(val_loss)
        # scheduler.step()
  

        writer.add_scalar('data/weighted_loss', weighted_train_loss, epoch)
        writer.add_scalars('data/loss', {
            'val loss': val_loss,
            'train loss': train_loss
        }, epoch)
        #记录数据
        # 记录到文本文件
        with open('training_log.txt', 'a') as f:
            f.write(f'Epoch {epoch} - Weighted Train Loss: {weighted_train_loss}\n')
            f.write(f'Epoch {epoch} - Val Loss: {val_loss}, Train Loss: {train_loss}\n')
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('--model', default="ghostnetv3", type=str)
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD
 
    # training 
    ##  -- optimizer
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- lr  
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)

    parser.add_argument('--end_epoch', default=500, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/snapshot/',  
                        type=str,
                        metavar='PATH')
    parser.add_argument('--log_file',
                        default="./checkpoint/train.logs",
                        type=str)
    parser.add_argument('--tensorboard',
                        default="./checkpoint/tensorboard",
                        type=str)
    parser.add_argument(
        '--resume',
        default='',
        type=str,
        metavar='PATH')

    # --dataset
    parser.add_argument('--dataroot',
                        default='./data/train_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--val_dataroot',
                        default='./data/test_data/list.txt',
                        type=str,
                        metavar='PATH')
    parser.add_argument('--train_batchsize', default=256, type=int)
    parser.add_argument('--val_batchsize', default=256, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
