import torch
import torch.nn as nn
import math
#方便调式，添加根目录
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_fastvit.modules.mobileone import MobileOneBlock
#初始化官方fastvit模型
import models
import models.model_fastvit


class Conv_Bloc(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(Conv_Bloc, self).__init__()

        #1×1卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        #3×3卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.adpavg=nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x=self.bn1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.adpavg(x)
        x=self.relu(x)
        return x


    


class FastVitPFLD(nn.Module):
    def __init__(self, backbone, num_classes,backbone_pretrained_path):
        super(FastVitPFLD, self).__init__()
        self.backbone_pretrained_path = backbone_pretrained_path
        self.cls_ratio=2
        self.inference_mode = False
        self.num_classes = num_classes
        if backbone == 'fastvit_t8':

            self.backbone = models.model_fastvit.fastvit_t8(pretrained=False,fork_feat=True,num_classes=self.num_classes)
            self.embed_dims = [48, 96, 192, 384]
        elif backbone == 'fastvit_t12':
            self.backbone = models.model_fastvit.fastvit_t12(pretrained=False,fork_feat=True,num_classes=self.num_classes)
            self.embed_dims = [64, 128, 256, 512]
        elif backbone == 'fastvit_s12':
            self.backbone = models.model_fastvit.fastvit_s12(pretrained=False,fork_feat=True,num_classes=self.num_classes)
            self.embed_dims = [64, 128, 256, 512]
        elif backbone == 'fastvit_sa12':    
            self.backbone = models.model_fastvit.fastvit_sa12(pretrained=False,fork_feat=True,num_classes=self.num_classes)
            self.embed_dims = [64, 128, 256, 512]
        elif backbone == 'fastvit_sa24':
            self.backbone = models.model_fastvit.fastvit_sa24(pretrained=False,fork_feat=True,num_classes=self.num_classes)
            self.embed_dims = [64, 128, 256, 512]

        else:
            raise Exception("模型不支持")
        #二分特征图header

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_exp = MobileOneBlock(
            in_channels=self.embed_dims[-1],
            out_channels=int(self.embed_dims[-1] * self.cls_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.embed_dims[-1],
            inference_mode=self.inference_mode,
            use_se=True,
            num_conv_branches=1,
        )
        self.head = (
            nn.Linear(int(self.embed_dims[-1] * self.cls_ratio), num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        
        for i in range(len(self.embed_dims)):
            self.add_module(f"Conv_Bloc{i+1}", Conv_Bloc(in_channels=self.embed_dims[i], out_channels=int(num_classes/2)))

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        # 遍历所有模块
        for m in self.modules():
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是线性层
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # 如果是批量归一化层
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        #直接加载预训练权重
        #加载权重文件
        pretrained_state_dict = torch.load(self.backbone_pretrained_path)
        #获取模型的字典
        model_dict = self.state_dict()
        # 按键值复制权重
        for key in pretrained_state_dict['state_dict']:
            if 'backbone.'+key in model_dict and pretrained_state_dict['state_dict'][key].shape == model_dict['backbone.'+key].shape:
                model_dict['backbone.'+key] = pretrained_state_dict['state_dict'][key]

  

    def forward(self, x):
        feat,x_cls = self.backbone(x)
        out_feat = []
        for i in range(len(feat)):
            # 根据索引动态获取对应的 Conv_Bloc 模块
            module_name = f"Conv_Bloc{i + 1}"
            conv_bloc_module = getattr(self, module_name)
            out_feat.append(conv_bloc_module(feat[i]))


        return out_feat,x_cls
    

if __name__=="__main__":
    model = FastVitPFLD('fastvit_t8', 98, r'D:\FILELin\postgraduate\小论文\疲劳驾驶检测\DFD\pfld\pretrain_model\fastvit_t8.pth.tar')
    x = torch.randn(1, 3, 224, 224)
    feat,x_cls = model(x)
    feat,x_cls = model(x)
    print(feat[0].shape)
    print(len(feat))