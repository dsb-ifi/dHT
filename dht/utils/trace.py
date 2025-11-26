import torch
import torch.nn.functional as F
import numpy as np

from typing import NamedTuple
from torch import Tensor
from io import StringIO

from .scatter import scatter_min_2d, scatter_mean_2d


def _find_all_bottom_left(seg:Tensor) -> Tensor:
    B, H, W = seg.shape
    flats = scatter_min_2d(
        torch.arange(B*H*W, device=seg.device).view(B,H,W).flip(1).view(-1), 
        seg.view(-1)
    )
    out = torch.stack(torch.unravel_index(flats, seg.shape), 0)
    out[1] = H - out[1] - 1
    return out


def _compute_probe_positions(curcoo:Tensor, dirs:Tensor) -> tuple[Tensor, Tensor]    :
    sub = dirs.diff(dim=0)[0]  # dirx - diry
    add = dirs.sum(dim=0)      # diry + dirx
    cs = F.pad(torch.stack([-sub-1, add-1], 0)//2, (0,0,1,0)).add_(curcoo)
    ds = F.pad(torch.stack([add-1, sub-1], 0)//2, (0,0,1,0)).add_(curcoo)
    return cs, ds


def _check_boundary_adherence(
    cs:Tensor, ds:Tensor, active_ids:Tensor, padseg:Tensor, H:int, W:int
) -> tuple[Tensor, Tensor]:
    # Check bounds
    c_valid = (cs[1] >= 0) & (cs[1] <= H) & (cs[2] >= 0) & (cs[2] <= W)
    d_valid = (ds[1] >= 0) & (ds[1] <= H) & (ds[2] >= 0) & (ds[2] <= W)
    
    # Check boundary adherence
    c = torch.zeros_like(c_valid)
    c[c_valid] = active_ids[c_valid] == padseg[cs[:,c_valid].unbind(0)]
    d = torch.zeros_like(d_valid)
    d[d_valid] = active_ids[d_valid] == padseg[ds[:,d_valid].unbind(0)]
    
    return c, d


def _apply_turns(dirs:Tensor, c:Tensor, d:Tensor) -> Tensor:
    right_turn = c
    left_turn = ~c & ~d
    
    newdirs = dirs.clone()
    
    # RIGHT turn: (diry, dirx) -> (-dirx, diry)
    newdirs[0, right_turn] = -dirs[1, right_turn]
    newdirs[1, right_turn] = dirs[0, right_turn]
    
    # LEFT turn: (diry, dirx) -> (dirx, -diry)
    newdirs[0, left_turn] = dirs[1, left_turn]
    newdirs[1, left_turn] = -dirs[0, left_turn]
    
    return newdirs


def trace_paths(seg:Tensor) -> Tensor:
    '''Trace paths along segmentation boundaries.

    Parameters
    ----------
    seg : Tensor
        Segmentation tensor of shape (B, H, W).
    
    Returns
    -------
    Tensor
        Traced paths tensor.
    '''
    B, H, W = seg.shape
    output = []
    
    # Find starting positions
    coos = _find_all_bottom_left(seg) + seg.new_tensor([[0],[1],[0]])
    n_regions = coos.shape[-1]
    
    # Initialize directions (always start going up)
    dirs = coos.new_tensor([[-1],[0]]).repeat(1, n_regions)
    
    # Pad segmentation for boundary checking
    padseg = F.pad(seg, (0,1,0,1), value=-1)
    
    # Initialize tracking
    curcoo = coos.clone()
    active_ids = torch.arange(n_regions, device=coos.device)
    
    # Record initial positions
    output.append(torch.cat([active_ids.unsqueeze(0), curcoo], 0))
    
    step = 1
    while True:
        # Move and check completion
        curcoo.add_(F.pad(dirs, (0,0,1,0)))        
        orig_for_active = coos[:, active_ids]
        just_completed = (curcoo[1] == orig_for_active[1]) & (curcoo[2] == orig_for_active[2])
        
        if just_completed.all():
            break
        
        # Filter completed paths
        still_active = ~just_completed
        curcoo = curcoo[:, still_active]
        dirs = dirs[:, still_active]
        active_ids = active_ids[still_active]
        
        cs, ds = _compute_probe_positions(curcoo, dirs)
        c, d = _check_boundary_adherence(cs, ds, active_ids, padseg, H, W)
        dirs = _apply_turns(dirs, c, d)
        
        # Record positions
        output.append(torch.cat([active_ids.unsqueeze(0), curcoo], 0))
        step += 1

    output = torch.cat(output, -1)
    output = output[:,output[0].argsort()]
    return output

