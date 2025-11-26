import torch
import torch.nn as nn
import os

from torch import Tensor
from .tok.tokenizer import dHTTokenizer

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

in1k_mean = torch.tensor([0.485, 0.456, 0.406])
in1k_std = torch.tensor([0.229, 0.224, 0.225])

def _in1k_norm(tensor:Tensor, dim:int=-1):
    shape = [1] * tensor.ndim
    shape[dim] = -1
    mean = in1k_mean.view(shape).to(tensor.device)
    std = in1k_std.reshape(shape).to(tensor.device)
    return (tensor - mean) / std


class dHTRas2Vec(nn.Module):
    def __init__(
        self, th0:float=0., th1:float=0.1, th2:float=0.1, 
        use_bg:bool=True, stroke:float=1.0, patch_size:int=32,
        cmp:float=0.01, iota:float=5.0, normalize:bool=False
    ):
        super().__init__()
        self.th0 = th0
        self.th1 = th1
        self.th2 = th2
        self.use_bg = use_bg
        self.stroke = stroke
        self.patch_size = patch_size
        self.normalize = normalize
        self.tokenizer = dHTTokenizer(3, 8, compute_grad=True, cmp=cmp, iota=iota)

    def forward(self, img: Tensor, device:torch.device=torch.device('cpu'), patch_size:int|None=None) -> str:
        if self.normalize:
            img = _in1k_norm(img, dim=1)
        patch_size = self.patch_size if patch_size is None else patch_size
        return self.tokenizer.ras2svg(img, th0=self.th0, th1=self.th1, th2=self.th2, 
            use_bg=self.use_bg, stroke=self.stroke, patch_size=self.patch_size,
            device=device
        )


def load_dht_ras2vec(
        pretrained:bool=True, th0:float=0., th1:float=0.1, th2:float=0.1, 
        use_bg:bool=True, stroke:float=1.0, patch_size:int=32,
        cmp:float=0.01, iota:float=5.0, normalize:bool=False
    ) -> dHTRas2Vec:
    
    if not pretrained:
        raise ValueError("Currently, only pretrained ras2vec models are available.")
    

    model = dHTRas2Vec(
        th0=th0, th1=th1, th2=th2, use_bg=use_bg, 
        stroke=stroke, patch_size=patch_size, cmp=cmp, iota=iota,
        normalize=normalize
    )

    if pretrained: # Always true for now
        path = os.path.join(__location__, '..', 'assets', 'r2v', 'r2v.pth')
        sd = torch.load(path, map_location='cpu')
        model.tokenizer.load_state_dict(sd)
    
    model.eval() 
    return model