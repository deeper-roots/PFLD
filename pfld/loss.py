import torch
from torch import nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle,
                landmarks, train_batchsize):
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        attributes_w_n = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        mat_ratio = torch.Tensor([
            1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio
        ]).to(device)
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)

        l2_distant = torch.sum(
            (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1)
        return torch.mean(weight_angle * weight_attribute *
                          l2_distant), torch.mean(l2_distant)
    

##多尺度热力图loss：点与heatmap混合结构的loss，可以参考：https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/modeling/losses/triplet_loss.py#L102
class MultiScaleFocalLoss(nn.Module):
    def __init__(self, input_size, num_classes, device):
        super(MultiScaleFocalLoss, self).__init__()
        self.input_size = input_size
        self.device = device
        self.num_classes = num_classes


    def forward(self, landmark_gt,
                landmarks_off,
                features_out,
                train_batchsize):
        #创建空的tensor
        landmarks=torch.zeros(features_out[0].shape[0],2*self.num_classes).to(self.device)

    # landmark 解算

        for j in range(4):
            # 在最后两个维度上查找最大值的索引
            max_indices = torch.argmax(features_out[j].view(features_out[j].shape[0], features_out[j].shape[1], -1), dim=-1)

            # 将一维索引转换为二维索引
            row_indices = max_indices // 2 # 在图像中表示x
            col_indices = max_indices % 2 #在图像中表示y
            landmarks[:,0:self.num_classes]+=row_indices.float()/(2**(j+1)) #在图像中表示x
            landmarks[:,self.num_classes:]+=col_indices.float()/(2**(j+1)) #在图像中表示y
        landmarks=landmarks+landmarks_off/(2**4)
        # 计算loss,landmarks与landmarks_gt的距离

        l2_distant = torch.mean(torch.sum(
            (landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1))

        return l2_distant

def smoothL1(y_true, y_pred, beta=1):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    mae = torch.abs(y_true - y_pred)
    loss = torch.sum(torch.where(mae > beta, mae - 0.5 * beta,
                                 0.5 * mae**2 / beta),
                     axis=-1)
    return torch.mean(loss)


def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK=106):
    y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
    y_true = y_true.reshape(-1, N_LANDMARK, 2)

    x = y_true - y_pred
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(w > absolute_x,
                         w * torch.log(1.0 + absolute_x / epsilon),
                         absolute_x - c)
    loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
    return loss