import torch
import torch.nn as nn
import torch.nn.functional as F


class Arcsinh(nn.Module):
    def __init__(self, in_ch, lmbda_init, learn_lmbda=True):
        super().__init__()
        if learn_lmbda:
            self.lmbda = nn.Parameter((lmbda_init*torch.ones(in_ch)).view(-1,1,1))
        else:
            self.register_buffer('lmbda', (lmbda_init*torch.ones(in_ch)).view(-1,1,1))

    def forward(self, x):
        m, d = self.lmbda, torch.arcsinh(self.lmbda)
        pos = self.lmbda > 0
        neg = self.lmbda < 0

        pos_output = x.mul(m).arcsinh().div(d)
        neg_output = x.mul(d).sinh().div(m)
        zro_output = x

        output = torch.where(pos, pos_output, torch.where(neg, neg_output, zro_output))
        return output
    

class HighBoost(nn.Module):

    def __init__(self, k=1.0, learnable=False):
        super().__init__()
        kernel = torch.ones(1,1,3,3)
        kernel[...,1,1] = 1 - kernel.sum()
        if learnable:
            self.k = nn.Parameter(k*torch.ones(1))
            self.kernel = nn.Parameter(kernel)
        else:
            self.register_buffer('k', k*torch.ones(1))
            self.register_buffer('kernel', kernel)

    def normalized_kernel(self):
        norm = self.kernel.abs().sum(dim=-1).sum(dim=-1)
        return self.kernel / norm[...,None,None]

    def forward(self, x):
        B, C, H, W = x.shape
        lap = F.conv2d(
            F.pad(x, (1,1,1,1), mode='replicate'), 
            self.normalized_kernel().expand(C,1,3,3), 
            groups=C
        )
        return x - self.k*lap


class GradOp(nn.Module):
    
    def __init__(self, learnable=False, **kwargs):
        super().__init__()
        kernel = torch.tensor([[[
            [-3., -10., -3.], 
            [ 0.,   0.,  0.], 
            [ 3.,  10.,  3.]
        ]]])
        if learnable:
            self.kernel = nn.Parameter(kernel)
        else:
            self.register_buffer('kernel', kernel)

    def normalized_kernel(self):
        norm = self.kernel.abs().sum(dim=-1).sum(dim=-1)
        return self.kernel / norm[...,None,None]

    def forward(self, x):
        kernel = self.normalized_kernel()
        return F.conv2d(
            F.pad(x.mean(dim=1, keepdim=True), 4*[1], mode='replicate'), 
            torch.cat([kernel, kernel.mT], 0)
        )

class SimpleDownsample(nn.Module):
    def __init__(self, in_ch, hid_ch, ratio=4.0):
        super().__init__()
        r = int(round(ratio))
        self.ds = nn.Sequential(
            nn.Conv2d(in_ch, hid_ch, 1, 1, 0),
            nn.BatchNorm2d(hid_ch),
        )
        self.cv = nn.Sequential(
            nn.Conv2d(in_ch, r*hid_ch, 2, 2),
            nn.BatchNorm2d(r*hid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(r*hid_ch, hid_ch, 2, 2),
            nn.BatchNorm2d(hid_ch),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.act = nn.Sequential(
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.act(
            self.ds(x) + self.cv(x)
        )