import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import NamedTuple, Optional
from .tokenizer import TokenizerResult

class dHTExtractorResult(NamedTuple):

    fV: Tensor
    seg: Tensor
    byx: Tensor
    nV: int


class InterpolationExtractor(nn.Module):

    def __init__(self, patch_size, channels, return_masks=True, normalize_interpolation=False):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.return_masks = return_masks
        grid_base = torch.linspace(0, 1, patch_size)
        self._vi = (-1, self.patch_size, self.patch_size, channels)
        self._vm = (-1, self.patch_size, self.patch_size, 1)
        ygrid, xgrid = torch.meshgrid(grid_base, grid_base, indexing='ij')
        self.normalize_interpolation = normalize_interpolation
        self.register_buffer('dims', torch.arange(end=channels), persistent=False)
        self.register_buffer('ygrid', ygrid.reshape(-1, self.patch_size**2, 1), persistent=False)
        self.register_buffer('xgrid', xgrid.reshape(-1, self.patch_size**2, 1), persistent=False)

    def forward(self, res:TokenizerResult):
        B, H, W = res.seg.shape
        C = res.fV.shape[-1]
        device = res.fV.device

        # Setup
        b = res.byx[0]
        b_idx = res.seg.view(-1).mul(B).add(b).unique() % B
        b_idx = b_idx.view(-1, 1, 1).expand(-1, self.patch_size**2, -1)
        c_idx = self.dims.view(1,1,-1).expand(*b_idx.shape[:2], -1) # type: ignore
        img = res.fV.view(B,H,W,-1)
        ymin, xmin, ymax, xmax = res.bb.view(4,-1,1,1)
        h_pos = self.ygrid * (ymax - ymin) + ymin
        w_pos = self.xgrid * (xmax - xmin) + xmin

        # Construct lower and upper bounds
        h_floor = h_pos.floor().long().clamp(0, H-1)
        w_floor = w_pos.floor().long().clamp(0, W-1)
        h_ceil = (h_floor + 1).clamp(0, H-1)
        w_ceil = (w_floor + 1).clamp(0, W-1)

        # Construct fractional parts of bilinear coordinates
        Uh, Uw = h_pos - h_floor, w_pos - w_floor
        Lh, Lw = 1 - Uh, 1 - Uw
        hfwf, hfwc, hcwf, hcwc = Lh*Lw, Lh*Uw, Uh*Lw, Uh*Uw

        # Get interpolated features
        bilinear = (
            img[b_idx, h_floor, w_floor, c_idx] * hfwf +
            img[b_idx, h_floor, w_ceil, c_idx] * hfwc +
            img[b_idx, h_ceil,  w_floor, c_idx] * hcwf +
            img[b_idx, h_ceil,  w_ceil, c_idx] * hcwc
        ).view(*self._vi).permute(0,3,1,2)

        if not self.return_masks:
            return bilinear

        # Get masks
        srange = torch.arange(b_idx.shape[0], device=device).view(-1,1)
        masks = (
            (res.seg[b_idx[:,:,0], h_floor[:,:,0], w_floor[:,:,0]] == srange).unsqueeze(-1) * hfwf +
            (res.seg[b_idx[:,:,0], h_floor[:,:,0], w_ceil[:,:,0]] == srange).unsqueeze(-1) * hfwc +
            (res.seg[b_idx[:,:,0], h_ceil[:,:,0], w_floor[:,:,0]] == srange).unsqueeze(-1) * hcwf +
            (res.seg[b_idx[:,:,0], h_ceil[:,:,0], w_ceil[:,:,0]] == srange).unsqueeze(-1) * hcwc
        ).view(*self._vm).permute(0,3,1,2)

        if self.normalize_interpolation:
            assert res.nV == masks.shape[0]
            bilinear = (
                bilinear * self.patch_size**2 / 
                masks.view(res.nV, -1).sum(-1).view(res.nV, 1, 1, 1).add_(1e-5)
            )
        return bilinear, masks
    

class PositionalHistogramExtractor(nn.Module):

    def __init__(self, patch_size:int, pos_patch_scale=None):
        super().__init__()
        self.patch_size = patch_size
        self.pos_patch_scale = pos_patch_scale
        if self.pos_patch_scale is None:
            self.pos_patch_scale = patch_size
    
    def forward(self, res:TokenizerResult, sizes:Optional[Tensor]=None):
        B, H, W = res.seg.shape
        if sizes is None:
            sizes = res.seg.view(-1).bincount()
        h_pos, w_pos = self.patch_size * res.byx[1:] / res.byx.new_tensor([[H], [W]])
        grid = res.fV.new_zeros(res.nV * self.patch_size**2)
        h_pos = h_pos.floor().long()
        w_pos = w_pos.floor().long()
        pos = res.seg.view(-1) * self.patch_size**2 + h_pos*self.patch_size + w_pos
        grid.scatter_add_(-1, pos, res.fV.new_ones(len(pos)))
        den = sizes.mul((self.pos_patch_scale/32)**2) # type: ignore
        return grid.view(res.nV, 1, self.patch_size, self.patch_size) / den.view(-1,1,1,1)


class GradientHistogramExtractor(nn.Module):

    def __init__(self, patch_size, eps=1e-7):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps

    def forward(self, res:TokenizerResult, sizes:Optional[Tensor]=None):
        if res.grad is None:
            return None
        if sizes is None:
            sizes = res.seg.view(-1).bincount()
        grad_y, grad_x = (
            self.patch_size * 
            res.grad.clip(self.eps-1, 1-self.eps)
                    .permute(1,0,2,3)
                    .reshape(2,-1)
                    .add(1)
                    .div(2)
        )
        grad = res.grad.new_zeros(res.nV * self.patch_size**2)
        grad_y = grad_y.floor().long()
        grad_x = grad_x.floor().long()
        pos = (
            res.seg.view(-1) * self.patch_size**2 + 
            grad_y*self.patch_size + grad_x
        )
        grad.scatter_add_(-1, pos, res.fV.new_ones(len(pos)))
        den = sizes.mul((self.patch_size/32)**2) # type: ignore
        return (
            grad.view(res.nV, 1, self.patch_size, self.patch_size) / 
            den.view(-1,1,1,1)
        )

class dHTExtractor(nn.Module):
    
    def __init__(
        self, patch_size, channels=3, mix_init=0.01, 
        learnable_mix=True,
        normalize_interpolation=False, 
        pos_patch_size=None, pos_patch_scale=None,
        **kwargs
    ):
        super().__init__()
        self.pos_patch_size = pos_patch_size if pos_patch_size is not None else patch_size
        self.patch_size = patch_size
        self.itp = InterpolationExtractor(patch_size, channels, True, normalize_interpolation, **kwargs)
        self.pos = PositionalHistogramExtractor(self.pos_patch_size, pos_patch_scale)
        self.grd = GradientHistogramExtractor(patch_size)
        mix = ((1/mix_init-1)*torch.ones(1)).log().neg()
        if learnable_mix:
            self._mix = nn.Parameter(mix)
        else:
            self.register_buffer("_mix", mix, persistent=True)
        dims = (channels, patch_size, patch_size)
        self.pixel_mask_token = nn.Parameter(1e-1 * torch.randn(*dims))
        self._store_masks = False
        self._masks = None

    def interpolate_pos_embed(self, new_pos_patch_size, pos_patch_scale=None):
        device = self._mix.device
        pos_patch_scale = pos_patch_scale if pos_patch_scale is not None else self.pos_patch_size
        self.pos = PositionalHistogramExtractor(new_pos_patch_size, pos_patch_scale).to(device)
        self.pos_patch_size = new_pos_patch_size

    @property
    def mix(self):
        return self._mix.sigmoid()
    
    @mix.setter
    def mix(self, val):
        self._mix.data = ((1/val-1)*torch.ones(1)).log().neg()

    def toggle_store_masks(self):
        self._store_masks = not self._store_masks

    @property
    def stored_mask(self):
        masks, self._masks = self._masks, None
        return masks

    def _features(self, res:TokenizerResult):
        ift, masks = self.itp(res)
        if self._store_masks:
            self._masks = masks
        mix = self._mix.sigmoid()
        invmask = 1 - masks
        return (
            ift * masks + 
            mix * invmask * ift +
            (1 - mix) * invmask * self.pixel_mask_token
        ).reshape(res.nV,-1)

    def _posgrad(self, res:TokenizerResult):
        pos = self.pos(res).view(res.nV,-1)
        grd = self.grd(res)
        if grd is not None:
            grd = grd.view(res.nV,-1)
        return torch.cat([pos] + ([] if grd is None else [grd]), -1)

        
    def forward(self, res:TokenizerResult):
        return dHTExtractorResult(
            torch.cat([self._features(res), self._posgrad(res)], -1), 
            res.seg, res.byx, res.nV
        )