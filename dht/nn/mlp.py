import torch.nn as nn

from .act import QGELU


class MLP(nn.Module):
    
    def __init__(self, embed_dim, hid_dim, actfn='gelu', **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.act = self.get_actfn(actfn)
        self.fc1 = nn.Linear(embed_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, embed_dim)

    @staticmethod
    def get_actfn(actstr):
        return {
            'gelu': nn.GELU(),
            'tgelu': nn.GELU('tanh'),
            'qgelu': QGELU(),
            'silu': nn.SiLU(),
        }.get(actstr, nn.GELU())
        
    def forward(self, x, *args, **kwargs):
        x = self.fc2(self.act(self.fc1(x)))
        return x
    

