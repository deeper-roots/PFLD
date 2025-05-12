#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import datetime
import logging
from pathlib import Path
import time
import os

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
def objective(trial,args):



    opt_name = trial.suggest_categorical(
        'opt_name', 
        [
            'Adam',      # 通用性强，适合大多数任务
            'AdamW',     # Adam + 解耦权重衰减（更适合Transformer类模型）
            'SGD',       # 配合动量使用，需精细调参
            'RMSprop',   # 适合RNN或非平稳目标
            'NAdam',     # Adam + Nesterov动量（收敛更快）
            'RAdam',     # 自适应学习率 + 收敛稳定性修正
            'Adagrad'    # 适合稀疏数据
        ]
    )

    # batchsize = trial.suggest_categorical('batchsize', [128, 256])
    batchsize = args.train_batchsize
    baselr = trial.suggest_float('baselr', 1e-4, 1e-2,step=0.0001)
    T_0 = trial.suggest_int('T_0', 10, 100)
    T_mult = trial.suggest_int('T_mult', 1, 10)
    #
    eta_min = trial.suggest_loguniform('eta_min', 1e-15, 1e-5)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-5)

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

    elif args.model == 'fastvit_s12':
        pfld_backbone = models.model_fastvit.fastvit_s12(pretrained=False,fork_feat=False,num_classes=196,
            ).to(device)
        auxiliarynet = models.mobilenetv4_gcon.AuxiliaryNet().to(device)

        #加载预训练权重
        #加载权重文件
        pretrained_state_dict = torch.load(r'./pretrain_model/fastvit_s12.pth.tar')
        #获取模型的字典
        model_dict = pfld_backbone.state_dict()
        # 按键值复制权重
        for key in pretrained_state_dict['state_dict']:
            if key in model_dict and pretrained_state_dict['state_dict'][key].shape == model_dict[key].shape:
                model_dict[key] = pretrained_state_dict['state_dict'][key]
    elif args.model == 'fastvit_t8':
        pfld_backbone = models.model_fastvit.fastvit_t8(pretrained=False,fork_feat=False,num_classes=196,
            ).to(device)
        auxiliarynet = models.mobilenetv4_gcon.AuxiliaryNet_input96().to(device)

        #加载预训练权重
        #加载权重文件    
        pretrained_state_dict = torch.load(r'./pretrain_model/fastvit_t8.pth.tar')
        #获取模型的字典
        model_dict = pfld_backbone.state_dict()
        # 按键值复制权重
        for key in pretrained_state_dict['state_dict']:
            if key in model_dict and pretrained_state_dict['state_dict'][key].shape == model_dict[key].shape:
                model_dict[key] = pretrained_state_dict['state_dict'][key]
    else:
        ##model not support
        raise Exception("模型不支持")
        
    #pfld_backbone = PFLDInference().to(device)
    # pfld_backbone = GhostStarNet().to(device)
    # auxiliarynet = AuxiliaryNet().to(device)
    criterion = PFLDLoss()
    ##默认使用adam优化器，已修改
    
    optimizer=getattr(torch.optim, opt_name)(
        [{'params': pfld_backbone.parameters()},
         {'params': auxiliarynet.parameters()}],
        lr=baselr,
        weight_decay=weight_decay
        )
    if args.lr_scheduler == 'cosinewr':
        #余弦热重启学习率调度器，带有热重启，容易跳出局部最优点
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
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
                            batch_size=batchsize,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=False)

    wlfw_val_dataset = WLFWDatasets(args.val_dataroot, transform)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset,
                                     batch_size=batchsize,
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
        # # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= args.early_stopping_patience:
        #         break

        #判断是否应该结束当前实验
        if trial.should_prune():
            raise optuna.TrialPruned(f'试验在 epoch {epoch} 被提前终止')        
    return best_val_loss


def main(args):
    # 超参自动搜索
    #存储器
    # 创建数据库存储
    storage_dir_path=args.snapshot.replace('snapshot','optuna_storage')   
    os.makedirs(storage_dir_path, exist_ok=True)  # 关键修改：添加exist_ok=True
    storage_path=os.path.join(storage_dir_path,'optuna.db')
    storage = RDBStorage(
        url=f"sqlite:///{storage_path}",  # 使用SQLite，也可以用MySQL/PgSQL
        heartbeat_interval=60,      # 心跳间隔，避免长时间运行时连接断开
        grace_period=120            # 优雅关闭时间
    )
    # 创建一个 Study 对象并开始优化
    # 定义 pruning 调度器
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # 在前 5 次试验中不进行 pruning
        n_warmup_steps=3,    # 每个试验在前 3 个 epoch 内不进行 pruning
        interval_steps=1,    # 每个 epoch 结束时检查一次是否应该 pruning
        n_min_trials=3      # 计算中位数的最小试验数量为 3
    )
    study = optuna.create_study(direction='minimize',
                                pruner=pruner,
                                storage=storage)
    # 使用 lambda 表达式来传递 args 参数
    study.optimize(lambda trial: objective(trial, args), n_trials=100)

    # 查看优化结果
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # 生成并保存可视化图表
    saved_path=args.snapshot.replace('snapshot','optuna_plots')
    os.makedirs(saved_path, exist_ok=True)
    
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(saved_path, "optimization_history.png"))
    
    fig = plot_param_importances(study)
    fig.write_image(os.path.join(saved_path, "param_importances.png"))
    
    fig = plot_parallel_coordinate(study)
    fig.write_image(os.path.join(saved_path, "parallel_coordinate.png"))

def parse_args():
    parser = argparse.ArgumentParser(description='pfld')
    # general
    parser.add_argument('--model', default="fastvit_s12", type=str)
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--devices_id', default='0', type=str)  #TBD
    parser.add_argument('--test_initial', default='false', type=str2bool)  #TBD
 
    # training 
    ##  -- optimizer
    #默认0.0001
    parser.add_argument('--base_lr', default=0.0001, type=int)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)
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
    args.model='fastvit_t8'

    args.end_epoch=30# 寻找最优超参的epoch
    args.lr_scheduler='cosinewr'

    #增加时间戳
    args.exp_name='fastvit_t8_'+str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    args.log_file='./exp_ho/'+args.model+'/'+args.exp_name+'/train.logs'
    args.tensorboard='./exp_ho/'+args.model+'/'+args.exp_name+'/tensorboard'
    args.snapshot='./exp_ho/'+args.model+'/'+args.exp_name+'/snapshot/'

    if not os.path.exists(args.snapshot):
        os.makedirs(args.snapshot)
    if not os.path.exists(args.tensorboard):
        os.makedirs(args.tensorboard)
    main(args)
