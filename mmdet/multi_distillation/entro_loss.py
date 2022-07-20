import torch
import torch.nn.functional as F
import torch.nn as nn


def torch_cov(x):
    h, w = x.size(2), x.size(3)
    a = x - torch.mean(x, dim=(2, 3), keepdim=True)
    a_T = a.permute(0, 1, 3, 2)
    cov_matrix = torch.matmul(a_T, a) / (h - 1)
    return cov_matrix


def Mahalanobis(feat_s, feat_t, mask):
    """
    feat_s: tensor: 1 * c * h * w
    feat_t: tensor: 1 * c * h * w
    """
    norm = max(1.0, mask.sum())
    dis = (feat_t - feat_s) * mask
    cov = torch_cov(dis)
    dis_T = dis.permute(0, 1, 3, 2)
    mah_dis = torch.abs(torch.matmul(torch.matmul(dis, cov), dis_T))
    mah_dis = torch.sqrt(mah_dis).sum() / norm
    return mah_dis


class entro_beta_loss(nn.Module):
    def __init__(self, batch_size, channel_size, init_num=2):
        super(entro_beta_loss, self).__init__()
        self.sigma = nn.Parameter(self.init_sigma(batch_size, channel_size, init_num))
        self.batch_size = batch_size
        self.channel_size = channel_size

    def init_sigma(self, batch_size, channel_size, init_num):
        sigma_w = torch.FloatTensor(torch.full(size=(batch_size, channel_size, 1, 1), fill_value=init_num))
        # sigma size batch_size * channel_size * 1 * 1
        return sigma_w

    def forward(self, teacher_fm, student_fm, mask, norms):
        """
        teacher_fm: batch_size * channel(512) * h * w
        """
        D_num = torch.pow(self.sigma, -2) * torch.pow(teacher_fm - student_fm, 2) * mask / norms
        sigma_loss = torch.log(torch.pow(self.sigma, 2))
        return (D_num + sigma_loss).sum() / (self.batch_size * self.channel_size)
