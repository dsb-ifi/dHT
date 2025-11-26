import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import NamedTuple, Optional

from .state import (
    TokenizerState, GridDimensions, 
    update_tokenizer_state, finalize_tokenization, token_drop_merge
)
from .modules import GradOp, Arcsinh, SimpleDownsample
from ..utils.ras2vec import ras2svg
from ..utils.segmentation import init_image_graph, get_seg_edges
from ..utils.scatter import scatter_mean_2d


class TokenizerResult(NamedTuple):
    '''NamedTuple holding tokenizer features and segmentation.

    Attributes
    ----------
    fV : Tensor
        Float tensor, pixel features of shape [BHW, C].
    seg : Tensor
        Segmentation tensor, mapping to original pixel locations.
    byx: Tensor
        Image coordinates.
    bb : Tensor
        Bounding box coordinates of regions.
    nV : int
        Number of vertices, regions.
    grad : Tensor, optional
        A tensor of image gradients.
    '''
    fV: Tensor
    seg: Tensor
    byx: Tensor
    bb: Tensor
    nV: int
    grad: Optional[Tensor] = None

    def to_svg(
        self, img:Tensor, th0:float=0., th1:float=0.1, th2:float=0.1, 
        use_bg:bool=True, image_index:int=0, stroke:float=1.0, patch_size:int=32,
        device:torch.device=torch.device('cpu')
    ) -> str:
        B, H, W = self.seg.shape
        return ras2svg(
            self.seg.to(device), img.to(device), th0=th0, th1=th1, th2=th2, 
            use_bg=use_bg, image_index=image_index, 
            stroke=stroke, patch_size=patch_size
        )


class dHTTokenizer(nn.Module):

    def __init__(
        self, in_ch, hid_ch, 
        similarity:str='gaussian', criterion:str='aicc',
        iota:float=5., eps:float=1e-8, cmp:float=0.1, center:float=0., 
        min_avg_region:int=6, compute_grad:bool=False,
        use_varproj:bool=True, inference_target_coeff:float=0.56,
        **kwargs
    ):
        super().__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.similarity = similarity
        self.criterion = criterion
        self.compute_grad = compute_grad
        self.iota = iota
        self.eps = eps
        self.cmp = cmp
        self.center = center
        self.min_avg_region = min_avg_region
        self.conv = SimpleDownsample(in_ch, hid_ch)
        self.linear = nn.Linear(hid_ch, in_ch)
        self.inference_target_coeff = 1/inference_target_coeff
        if compute_grad:
            self.gradact = Arcsinh(2, 0, learn_lmbda=True)
            self.gradient = GradOp(k=2.0, learnable=False)
        else:
            self.gradact = nn.Identity()
            self.gradient = nn.Identity()
        self.varproj = None
        if use_varproj:
            self.varproj = nn.Linear(hid_ch, 1)
            self.varproj.weight.data = torch.randn(1, hid_ch) * 1e-4
            self.varproj.bias.data = -torch.ones(1)*2*torch.pi
    
    def preproc(self, img) -> tuple[Tensor, Optional[Tensor]]:
        if self.compute_grad:
            grad = self.gradact(self.gradient(img))
            return self.conv(img).permute(0,2,3,1).reshape(-1, self.hid_ch), grad
        return self.conv(img).permute(0,2,3,1).reshape(-1, self.hid_ch), None
      
    def expand_with_grad(self):
        self.gradact = Arcsinh(2, 27.5, learn_lmbda=True)
        self.gradient = GradOp(k=2.0, learnable=False)
        self.compute_grad = True
        
    def init_tokenizer_state(self, img):
        B,C,H,W = img.shape
        V, E, mV, nV, byx = init_image_graph(img)
        bb = torch.stack([byx[1],byx[2],byx[1],byx[2]], 0)
        fV, grad = self.preproc(img)
        s2 = torch.zeros_like(fV)
        info = torch.full_like(fV[:,0], -(H*W)**.5)
        curseg = V
        optnV = nV
        optinfo = info
        optseg = V
        optfV = fV
        opts2 = s2
        grid = GridDimensions(byx, B, C, H, W)
        state = TokenizerState(
            fV, s2, E, mV, info, curseg, optseg, optinfo, optfV, opts2,
            nV, optnV, grid, bb
        )
        return state, grad
    
    def mean_injection(self, img, res):
        fV = img.permute(0,2,3,1).reshape(-1, img.shape[1])
        replaced_mean = res.fV - scatter_mean_2d(fV, res.seg)
        return fV + replaced_mean[res.seg]
    
    def _refactor(self, res:TokenizerResult):
        B, H, W = res.seg.shape
        x = res.fV
        concat = 2*res.byx[1:].mT / x.new_tensor([H-1,W-1]) - 1
        if res.grad is not None:
            concat = torch.cat([
                concat, res.grad.permute(0,2,3,1).reshape(B*H*W, -1)
            ], -1)
        sizes = res.seg.view(-1).bincount()
        return torch.cat([x, concat], -1), None, res.seg, res.byx, sizes, res.nV, res.bb
            
    
    def forward(
        self, img, pretrain=False, final_merging=True, target=None, 
        max_it_override=None, residuals=False,
    ):
        if target is None and not self.training:
            target = int(round(self.inference_target_coeff * (img.shape[-2]*img.shape[-1])**.5))

        state, grad = self.init_tokenizer_state(img)
        origE = state.E
        it = 0
        max_it = (
            int(np.ceil(np.log2((state.grid.H * state.grid.W)**.5)))
            if max_it_override is None else max_it_override
        )
        stop = False
        while (state.nV / state.grid.B > self.min_avg_region) and it < max_it and not stop:
            state, stop = update_tokenizer_state(
                state, it, self.similarity, self.criterion, 
                self.cmp, self.center, self.iota, self.eps,
                self.varproj
            )
            it += 1
        
        res = finalize_tokenization(
            state, origE, self.criterion, self.iota, self.eps,
            self.linear
        )

        if final_merging:
            res = token_drop_merge(res, similarity=self.similarity, target=target)

        if pretrain:
            return res

        B, H, W = res.grid.B, res.grid.H, res.grid.W

        if residuals:
            residuals = img.permute(0,2,3,1) - scatter_mean_2d(
                img.permute(0,2,3,1).reshape(-1,img.shape[1]), 
                res.seg.view(-1)
            )[res.seg.view(B,H,W)]
            return residuals

        return TokenizerResult(
            self.mean_injection(img, res),
            res.seg.view(B,H,W), res.grid.byx, res.bb, res.nV, grad
        )
    
    def ras2svg(
        self, img:Tensor, th0:float=0., th1:float=0.1, th2:float=0.1, 
        use_bg:bool=True, image_index:int=0, stroke:float=1.0, patch_size:int=32,
        device:torch.device=torch.device('cpu')
    ) -> str:
        B,C,H,W = img.shape
        target = int(round((H*W)**.5))
        with torch.no_grad():
            forward_res = self.forward(img, pretrain=True, target=target, final_merging=True)
        
        return forward_res.to_svg(
            img.to(device), th0=th0, th1=th1, th2=th2, 
            use_bg=use_bg, image_index=image_index, 
            stroke=stroke, patch_size=patch_size
        )