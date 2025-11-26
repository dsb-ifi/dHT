import torch
import torch.nn as nn

class QGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)