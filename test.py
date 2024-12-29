# ------------------------------------------------------------------------------
# Copyright (c) Zhichao Zhao
# Licensed under the MIT License.
# Created by Zhichao zhao(zhaozhichao4515@gmail.com)
# ------------------------------------------------------------------------------
import argparse
import time

import cv2
import numpy as np

from matplotlib import pyplot as plt
from scipy.integrate import simps
import  yaml

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import WLFWDatasets

import models.pfld
import models.ghostnetv3
import models.mobilenetv4
import models.mobilenetv4_gcon


cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device( "cpu")
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt,
                                        axis=1)) / (interocular * L)

    return rmse


def compute_auc(errors, failureThreshold, step=0.0001, showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate


def validate(wlfw_val_dataloader, pfld_backbone):
    pfld_backbone.eval()

    nme_list = []
    cost_time = []
    with torch.no_grad():
        for img, landmark_gt, _, _ in wlfw_val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)

            start_time = time.time()
            _, landmarks = pfld_backbone(img)
            cost_time.append(time.time() - start_time)

            landmarks = landmarks.cpu().numpy()
            landmarks = landmarks.reshape(landmarks.shape[0], -1,
                                          2)  # landmark
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1,
                                              2).cpu().numpy()  # landmark_gt

            if args.show_image:
                show_img = np.array(
                    np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
                show_img = (show_img * 255).astype(np.uint8)
                np.clip(show_img, 0, 255)

                pre_landmark = landmarks[0] * [112, 112]

                cv2.imwrite("show_img.jpg", show_img)
                img_clone = cv2.imread("show_img.jpg")

                for (x, y) in pre_landmark.astype(np.int32):
                    cv2.circle(img_clone, (x, y), 1, (255, 0, 0), -1)
                cv2.imshow("show_img.jpg", img_clone)
                cv2.waitKey(0)

            nme_temp = compute_nme(landmarks, landmark_gt)
            for item in nme_temp:
                nme_list.append(item)

        # nme
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        # auc and failure rate
        failureThreshold = 0.1
        auc, failure_rate = compute_auc(nme_list, failureThreshold)
        print('auc @ {:.1f} failureThreshold: {:.4f}'.format(
            failureThreshold, auc))
        print('failure_rate: {:}'.format(failure_rate))
        # inference time
        print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))
        

def main(args):
    checkpoint = torch.load(args.model_path, map_location=device)

    #pfld_backbone = PFLDInference().to(device)
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
        未修改baobone，仅仅删除了一个下采样层，推理速度，后期模型与一开始迭代的模型翻倍（偶发情况），精度比上一个高，采用原始的mobilenetV4检测头
        loss能达到0.2011,推理速度5.6ms
        '''
        pfld_backbone = models.mobilenetv4_gcon.mobilenetv4_conv_small_ori1().to(device)
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
    
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'],strict=False)

    transform = transforms.Compose([transforms.ToTensor()])
    wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=0)
    pfld_backbone.eval()

    validate(wlfw_val_dataloader, pfld_backbone)
    #export onnx 
    # img = torch.randn(1, 3, 112, 112).to(device)
    # torch.onnx.export(pfld_backbone, img, "./result_model/pfld_backbone.onnx", verbose=True, opset_version=11)
    
    

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model', default="mobilenetv4_ori", type=str)
    parser.add_argument('--model_path',
                        default="./checkpoint/snapshot/checkpoint_epoch_2.pth.tar",   #"./checkpoint/ghostnet/Ghostnet/11_3/checkpoint_epoch_164.pth.tar",#"./checkpoint/ghostnet/PFLD原/checkpoint_epoch_263.pth.tar",        #"./checkpoint/ghostnet/Ghostnet/11_3/checkpoint_epoch_164.pth.tar",
                        type=str)
    parser.add_argument('--test_dataset',
                        default='./data/test_data/list.txt',
                        type=str)
    parser.add_argument('--show_image', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
