import torch
import torch.nn.functional as F
import numpy as np

from typing import NamedTuple
from torch import Tensor
from io import StringIO
from .trace import trace_paths


class PiecewiseLinearPaths(NamedTuple):
    ids:Tensor
    b:Tensor
    y:Tensor
    x:Tensor

    @classmethod
    def from_segmentation(cls, seg):
        ids, b, y, x = trace_paths(seg)
        return cls(ids, b, y.float(), x.float())

    def vw_compute_areas(self) -> Tensor:
        '''Visvalingam-Whyatt area computation.

        Returns
        -------
        Tensor
            Tensor (float) of areas.
        '''
        bc = self.ids.bincount()
        B = bc.cumsum(-1)
        A = B - bc
        lens = bc[self.ids]
        starts = A[self.ids]
        stops = B[self.ids]

        idxs = torch.arange(len(lens), device=self.ids.device)
        locid = idxs - starts
        wins = torch.stack([(locid - 1) % lens, locid, (locid + 1) % lens], 0) + starts[None].expand(3,-1)
        x, y = self.x, self.y
        return (
            x[wins[0]]*y[wins[1]] + x[wins[1]]*y[wins[2]] + x[wins[2]]*y[wins[0]] - 
            x[wins[0]]*y[wins[2]] - x[wins[1]]*y[wins[0]] - x[wins[2]]*y[wins[1]]
        ).abs()

    def simplify(self, th:float=0.) -> "PiecewiseLinearPaths":
        '''Simplify paths using greedy Visvalingam-Whyatt algorithm.

        Parameters
        ----------
        th : float, optional
            Area threshold for simplification, by default 0.
        
        Returns
        -------
        "VPaths"
            Simplified paths.
        '''
        areas = self.vw_compute_areas()
        mask = areas > th
        return self.__class__(self.ids[mask], self.b[mask], self.y[mask], self.x[mask])

    def chaikin(self) -> "PiecewiseLinearPaths":
        '''Apply Chaikin's corner-cutting algorithm to smooth paths.

        Returns
        -------
        "VPaths"
            Smoothed paths.
        '''
        regions = self.ids
        y, x = self.y, self.x
        
        bc = regions.bincount()
        B = bc.cumsum(0)
        A = B - bc
        lens   = bc[regions]
        starts = A[regions]
        idx = torch.arange(len(regions), device=regions.device)
        loc = idx - starts
        loc_next = (loc + 1) % lens
        g_next   = starts + loc_next
        
        Qx = 0.75*x[idx] + 0.25*x[g_next]
        Qy = 0.75*y[idx] + 0.25*y[g_next]
        Rx = 0.25*x[idx] + 0.75*x[g_next]
        Ry = 0.25*y[idx] + 0.75*y[g_next]
        
        new_bc = bc * 2
        new_B = new_bc.cumsum(0)
        new_A = new_B - new_bc
        out_Q = new_A[regions] + 2*loc
        out_R = out_Q + 1
        total = int(new_bc.sum())
        new_x = torch.empty(total, device=regions.device, dtype=x.dtype)
        new_y = torch.empty_like(new_x)
        
        new_x[out_Q] = Qx
        new_y[out_Q] = Qy
        new_x[out_R] = Rx
        new_y[out_R] = Ry

        new_ids = regions.repeat_interleave(2)
        new_b = self.b.repeat_interleave(2)
        
        return self.__class__(new_ids, new_b, new_y, new_x)


def _beziercurves_to_svg(
    curves:"BezierCurves", region_colors:Tensor, 
    width:int, height:int, stroke:float=1.0, 
    buf:StringIO|None=None, finalize:bool=True
) -> StringIO:
    ids = curves.ids
    cp = curves.cp
    
    bc = ids.bincount().cpu().numpy()
    boundaries = np.concatenate([[0], np.cumsum(bc)])
    R = len(bc)
    
    cp_np = cp.cpu().numpy()
    
    # Colors
    if isinstance(region_colors, torch.Tensor):
        colors_np = (region_colors * 255).round().long().cpu().numpy()
        color_strs = [f"rgb({c[0]},{c[1]},{c[2]})" for c in colors_np]
    else:
        color_strs = region_colors
    
    # Pre-format all cubic commands at once
    N = cp_np.shape[2]
    cubic_cmds = []
    for i in range(N):
        cubic_cmds.append(
            f'C {cp_np[1,0,i]:.3f},{cp_np[1,1,i]:.3f} '
            f'{cp_np[2,0,i]:.3f},{cp_np[2,1,i]:.3f} '
            f'{cp_np[3,0,i]:.3f},{cp_np[3,1,i]:.3f}'
        )

    if buf is None:
        buf = StringIO()
        buf.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n')
    
    for r in range(R):
        start, end = boundaries[r], boundaries[r+1]
        if start == end:
            continue
        
        buf.write(f'<path d="M {cp_np[0,0,start]:.3f},{cp_np[0,1,start]:.3f}')
        
        # Write each cubic command directly (no join overhead)
        for i in range(start, end):
            buf.write(' ')
            buf.write(cubic_cmds[i])
        
        buf.write(f' Z" stroke="{color_strs[r]}" stroke-width="{stroke}" fill="{color_strs[r]}"/>\n')

    if finalize:
        buf.write('</svg>')
    
    return buf


def _write_patch_background(
    image:Tensor, patch_size:int, buf:StringIO|None=None, finalize:bool=False
) -> StringIO:
    C, H, W = image.shape
    assert C == 3
    
    # Downsample using nearest neighbor to get patch colors
    patches = F.interpolate(
        image.unsqueeze(0),
        scale_factor=1/patch_size,
        mode='nearest'
    ).squeeze(0).permute(1, 2, 0)
    
    patch_h, patch_w = patches.shape[:2]
    patches_np = (patches * 255).round().long().cpu().numpy()
    
    # Build coordinate grids vectorized
    px_grid, py_grid = np.meshgrid(np.arange(patch_w), np.arange(patch_h), indexing='xy')
    x = (px_grid * patch_size).flatten()
    y = (py_grid * patch_size).flatten()
    w = np.minimum(patch_size, W - x)
    h = np.minimum(patch_size, H - y)
    
    # Flatten colors
    r = patches_np[:, :, 0].flatten()
    g = patches_np[:, :, 1].flatten()
    b = patches_np[:, :, 2].flatten()
    
    # Stack into (N, 7) array: x, y, w, h, r, g, b
    rect_data = np.column_stack([x, y, w, h, r, g, b])

    if buf is None:
        buf = StringIO()
        buf.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">\n')
    
    # Write with savetxt using format per column
    np.savetxt(
        buf,
        rect_data,
        fmt='<rect x="%d" y="%d" width="%d" height="%d" fill="rgb(%d,%d,%d)"/>',
        delimiter='',
        newline='\n',
        comments=''
    )

    if finalize:
        buf.write('</svg>')

    return buf


class BezierCurves(NamedTuple):
    ids: Tensor
    b: Tensor
    cp: Tensor

    @classmethod
    def from_pwl(cls, vp:PiecewiseLinearPaths, stride:int=4, alpha:float=0.33):
        ids = vp.ids
        x, y = vp.x, vp.y
        b     = vp.b
    
        bc = ids.bincount()
        B  = bc.cumsum(0)
        A  = B - bc
    
        idx = torch.arange(len(ids), device=ids.device)
        lens   = bc[ids]
        starts = A[ids]
        loc    = idx - starts
    
        # sample segment start points
        mask = (loc % stride) == 0
    
        seg_idx     = idx[mask]
        seg_ids     = ids[mask]
        seg_b       = b[mask]
        seg_loc     = loc[mask]
        seg_starts  = starts[mask]
        seg_lens    = lens[mask]
    
        # circular neighbor indexer
        def circ(off):
            return seg_starts + ((seg_loc + off) % seg_lens)
    
        # endpoints
        idx0 = seg_idx
        idx3 = circ(stride)
    
        # tangents
        idx_prev0 = circ(-1)
        idx_next0 = circ(+1)
        idx_prev3 = circ(stride - 1)
        idx_next3 = circ(stride + 1)
    
        # unit tangents
        t0x = x[idx_next0] - x[idx_prev0]
        t0y = y[idx_next0] - y[idx_prev0]
        t3x = x[idx_next3] - x[idx_prev3]
        t3y = y[idx_next3] - y[idx_prev3]
    
        # normalize
        n0 = (t0x*t0x + t0y*t0y).sqrt().clamp_min(1e-6)
        n3 = (t3x*t3x + t3y*t3y).sqrt().clamp_min(1e-6)
        t0x, t0y = t0x/n0, t0y/n0
        t3x, t3y = t3x/n3, t3y/n3
    
        # chord length
        dx = x[idx3] - x[idx0]
        dy = y[idx3] - y[idx0]
        chord = (dx*dx + dy*dy).sqrt()
    
        a = alpha * chord
    
        # bezier control points
        P0x, P0y = x[idx0], y[idx0]
        P3x, P3y = x[idx3], y[idx3]
    
        P1x = P0x + a * t0x
        P1y = P0y + a * t0y
        P2x = P3x - a * t3x
        P2y = P3y - a * t3y
    
        cp = torch.stack([
            torch.stack([P0x, P0y], 0),
            torch.stack([P1x, P1y], 0),
            torch.stack([P2x, P2y], 0),
            torch.stack([P3x, P3y], 0)
        ], 0)
    
        return cls(seg_ids, seg_b, cp)
    

    def to_svg(
        self, region_colors:Tensor, stroke:float=1.0, 
        bg_image:Tensor|None=None, patch_size:int=32,
        image_index:int=0, width:int|None=None, height:int|None=None
    ) -> StringIO:
        '''Convert Bezier curves to SVG format.

        Parameters
        ----------
        region_colors : Tensor
            Tensor of shape (R, 3) with RGB colors for each region.
        stroke : float, optional
            Stroke width for the paths, by default 1.0.
        bg_image : Tensor | None, optional
            Background image tensor of shape (C, H, W), by default None.
        patch_size : int, optional
            Patch size for background image, by default 32.
        image_index : int, optional
            Index of the image in a batch if bg_image is batched, by default 0.
        width : int | None, optional
            Width of the SVG canvas, by default None.
        height : int | None, optional
            Height of the SVG canvas, by default None.

        Returns
        -------
        StringIO
            StringIO buffer containing the SVG data.
        '''
        buf = None
        if bg_image is not None:
            if bg_image.ndim == 4:
                img = bg_image[image_index]
            else:
                img = bg_image
            C, H, W = img.shape
            assert C == 3, f'Background image must have 3 channels, got {C}!'
            if width is None:
                width = W
            if height is None:
                height = H
            buf = _write_patch_background(img, patch_size, buf=None, finalize=False)
        else:
            if width is None or height is None:
                raise ValueError('Width and height must be specified if no background image is provided.')

        buf = _beziercurves_to_svg(
            self, region_colors, width, height, 
            stroke=stroke, buf=buf, finalize=True
        )
        return buf