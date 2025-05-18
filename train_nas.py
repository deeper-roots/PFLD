#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import datetime
import logging
from pathlib import Path
import time
import os

import nni
import numpy as np
import optuna
import torch

from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,StepLR

import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from dataset.datasets import WLFWDatasets
import models.models_NAS
import models.pfld
import models.ghostnetv3
import models.mobilenetv4
import models.mobilenetv4_gcon
import  models.model_fastvit
#from models.pfld import PFLDInference, AuxiliaryNet
#from models.ghostnetv3 import GhostStarNet,AuxiliaryNet
from pfld.loss import PFLDLoss
from pfld.utils import AverageMeter

from optuna.storages import RDBStorage
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)

import nni
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear,ValueChoice
from nni.nas.experiment import NasExperiment

import matplotlib
matplotlib.use('Agg')  # 必须在导入 matplotlib.pyplot 之前设置
import matplotlib.pyplot as plt

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






#optuna 超参优化
def evaluate_model(model,args):



    # Step 1: data
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
        auxiliarynet = models.pfld.AuxiliaryNet().to(device)
    elif args.model == 'ghostnetv3':
        auxiliarynet = models.ghostnetv3.AuxiliaryNet().to(device)
    elif args.model =='mobilenetv4_small_OriHeater':
        '''
        该模型并非原始模型，推理速度，精度整体符合要求，略低于PFLD
        该模型原本输入224，下载输入112，导致stage对应的size不太一样
        
        '''
        auxiliarynet = models.mobilenetv4.AuxiliaryNet().to(device)
    elif args.model =='mobilenetv4_small_bockbone_fixed':
        '''
        该模型仅仅改变backbone的参数，调整了以下stage对应的pix size,推理速度5ms，比原版mobilenetv4_conv_small慢了一秒
        准确率8.9，与pfld少了两个点
        '''
        auxiliarynet = models.mobilenetv4.AuxiliaryNet().to(device)
    elif args.model =='mobilenetv4_ori':
        '''
        mobilenetv4_conv_small_ori
        未修改baobone，仅仅删除了一个下采样层，推理速度，后期模型与一开始迭代的模型翻倍（偶发情况），精度比上一个高，采用原始的mobilenetV4检测头
        loss能达到0.2011,推理速度5.6ms
        mobilenetv4_conv_small_ori1,loss差不多，但是推理速度慢很多，8.9ms
        '''
        auxiliarynet = models.mobilenetv4_gcon.AuxiliaryNet_Ori().to(device)
    elif args.model =='mobilenetv4_gcon':
        '''
        原始模型，采用gcon检测头，

        '''
        auxiliarynet = models.mobilenetv4_gcon.AuxiliaryNet_Ori().to(device)

    elif args.model == 'fastvit_s12':
        auxiliarynet = models.mobilenetv4_gcon.AuxiliaryNet().to(device)

    elif args.model == 'fastvit_t8_nas':

        auxiliarynet = models.mobilenetv4_gcon.AuxiliaryNet_input96().to(device)

    else:
        ##model not support
        raise Exception("模型不支持")
        
    pfld_backbone=model

    criterion = PFLDLoss()
    ##默认使用adam优化器，已修改
    
    optimizer=getattr(torch.optim, args.opt_name)(
        [{'params': pfld_backbone.parameters()},
         {'params': auxiliarynet.parameters()}],
        lr=args.base_lr,
        weight_decay=args.weight_decay
        )
    if args.lr_scheduler == 'cosinewr':
        #余弦热重启学习率调度器，带有热重启，容易跳出局部最优点
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
    elif args.lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, verbose=True)
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        #该学习率调度器 效果不太好
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=args.lr_patience, verbose=True)
    else:
        raise Exception("lr_scheduler not support")
    

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])
    wlfwdataset = WLFWDatasets(args.dataroot, transform)
    dataloader = DataLoader(wlfwdataset,
                            batch_size=args.batchsize,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=False)

    wlfw_val_dataset = WLFWDatasets(args.val_dataroot, transform)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset,
                                     batch_size=args.batchsize,
                                     shuffle=False,
                                     num_workers=args.workers)

    # step 4: run
    best_val_loss=float('inf')
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        weighted_train_loss, train_loss = train(dataloader, pfld_backbone,
                                                auxiliarynet, criterion,
                                                optimizer,
                                                 epoch)

        val_loss = validate(wlfw_val_dataloader, pfld_backbone, auxiliarynet,
                            criterion)
        
        #根据损失设置学习率
        if args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        logging.info(
            'epoch: {}/{}, weighted_train_loss: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}'
            .format(epoch, args.end_epoch, weighted_train_loss, train_loss,
                    val_loss))
        #
        nni.report_intermediate_result(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss

     
    nni.report_final_result(val_loss)

def evaluator_function(model):
    return evaluate_model(model, args)

def main(args):
    # nas自动搜索
  
    #模型初始化
    if args.model == 'fastvit_s12':
        pfld_backbone = models.model_fastvit.fastvit_s12(pretrained=False,fork_feat=False,num_classes=196,
            ).to(device)

    elif args.model == 'fastvit_t8_nas':
        pfld_backbone_searchspace = models.models_NAS.fastvit_t8_nas(pretrained=False,fork_feat=False,num_classes=196,
            ).to(device)
    else:
        ##model not support
        raise Exception("模型不支持")


    #创建评估器
    evaluator = FunctionalEvaluator(evaluator_function)
    #搜索算法
    import nni.nas.strategy as strategy
    search_strategy = strategy.Random()  # dedup=False if deduplication is not wanted
    #创建实验
    exp = NasExperiment(pfld_backbone_searchspace, evaluator, search_strategy)
    exp.config.experiment_working_directory = args.snapshot
    # exp.config.training_service.checkpoint_dir = args.snapshot + "/checkpoint"
    exp.config.trial_gpu_number = args.trial_gpu_number
    exp.config.training_service.use_active_gpu = args.use_active_gpu
    
    exp.run(port=8082)

def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('--model', default="fastvit_s12", type=str)
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD
 
    # training 
    ##  -- optimizer
    parser.add_argument('--opt_name', default='Adam', type=str)
    #默认0.0001
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight_decay', '--wd', default=1e-6, type=float)
    #学习率调度器参数
    ## 学习率调度器类型
    parser.add_argument('--lr_scheduler', default='step', type=str)
    ## step学习率调度器默认参数
    parser.add_argument('--step_size', default=40, type=int)#step学习率调度器的step的大小，多少个epoches后减少学习率
    parser.add_argument('--gamma', default=0.1, type=float)# 默认的学习率衰减率

    # 余弦退火学习率调度器默认参数
    parser.add_argument('--T_0', default=50, type=int)
    parser.add_argument('--T_mult', default=1, type=int)
    parser.add_argument('--eta_min', default=1e-10, type=float)
    # -- 自动调整学习率调度器默认参数 lr  
    parser.add_argument("--lr_patience", default=40, type=int)

    # -- epoch
    parser.add_argument('--start_epoch', default=1, type=int)

    parser.add_argument('--end_epoch', default=500, type=int)
    #nas搜索参数
    parser.add_argument('--search_strategy', default='random', type=str)
    parser.add_argument('--trial_concurrency', default=10, type=int)
    parser.add_argument('--max_trial_number', default=50, type=int)
    parser.add_argument('--trial_gpu_number', default=1, type=int)
    parser.add_argument('--use_active_gpu', default=True, type=bool)

    parser.add_argument('--population_size', default=10, type=int)
    parser.add_argument('--sample_size', default=1, type=int)   
    parser.add_argument('--mutation_rate', default=0.1, type=float)
    parser.add_argument('--crossover_rate', default=0.1, type=float)

    #早停设置
    parser.add_argument('--early_stopping_patience',
                        default=10,  
                        type=int)
                        
    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot',
                        default='./checkpoint/snapshot/',  
                        type=str,
                        metavar='PATH')  #保存模型的路径
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
    args.model='fastvit_t8_nas'

    args.end_epoch=30# 寻找最优超参的epoch
    args.lr_scheduler='cosinewr'

    #增加时间戳
    args.exp_name=args.model+'_'+str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    exp_pre='./exp_nas/'
    args.log_file=exp_pre+args.model+'/'+args.exp_name+'/train.logs'
    args.tensorboard=exp_pre+args.model+'/'+args.exp_name+'/tensorboard'
    args.snapshot=exp_pre+args.model+'/'+args.exp_name+'/snapshot/'

    if not os.path.exists(args.snapshot):
        os.makedirs(args.snapshot)
    if not os.path.exists(args.tensorboard):
        os.makedirs(args.tensorboard)
    main(args)
