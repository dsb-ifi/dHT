import torch
from torch import Tensor

from .curves import PiecewiseLinearPaths, BezierCurves
from .scatter import scatter_mean_2d

in1k_mean = torch.tensor([0.485, 0.456, 0.406])
in1k_std = torch.tensor([0.229, 0.224, 0.225])

def _in1k_unnorm(tensor:Tensor, dim=-1):
    shape = [1] * tensor.ndim
    shape[dim] = -1
    mean = in1k_mean.view(shape).to(tensor.device)
    std = in1k_std.reshape(shape).to(tensor.device)
    return tensor * std + mean


def ras2bezier(seg:Tensor,th0:float=0., th1:float=0.1, th2:float=0.1) -> BezierCurves:
    '''Convert raster segmentation to Bezier curves.

    Parameters
    ----------
    seg : Tensor
        Segmentation tensor of shape (B,H,W) with integer segment IDs.
    th0 : float, optional
        Simplification threshold for initial simplification, by default 0.
    th1 : float, optional
        Simplification threshold for second simplification, by default 0.1.
    th2 : float, optional
        Simplification threshold for final simplification, by default 0.1.

    Returns
    -------
    BezierCurves
        Bezier curves representing the segmentation boundaries.
    '''
    paths = PiecewiseLinearPaths.from_segmentation(seg)
    paths = paths.simplify(th=th0)
    paths = paths.chaikin()
    paths = paths.simplify(th=th1)
    paths = paths.chaikin()
    paths = paths.simplify(th=th2)
    return BezierCurves.from_pwl(paths)


def ras2svg(
    seg:Tensor, img:Tensor, th0:float=0., th1:float=0.1, th2:float=0.1, 
    use_bg:bool=True, image_index:int=0, stroke:float=1.0, patch_size:int=32
) -> str:
    '''Convert raster segmentation to SVG representation.

    Parameters
    ----------
    seg : Tensor
        Segmentation tensor of shape (B,H,W) with integer segment IDs.
    img : Tensor
        Image tensor of shape (B,3,H,W) with values in [0,1].
    th0 : float, optional
        Simplification threshold for initial simplification, by default 0.
    th1 : float, optional
        Simplification threshold for second simplification, by default 0.1.
    th2 : float, optional
        Simplification threshold for final simplification, by default 0.1.

    Returns
    -------
    str
        SVG representation of the segmentation boundaries.
    '''
    B,H,W = seg.shape
    if B > 1:
        seg = seg[image_index:image_index+1]
        seg = seg - seg.min()
        img = img[image_index:image_index+1]

    img = _in1k_unnorm(img, 1)
        
    bezier = ras2bezier(seg, th0=th0, th1=th1, th2=th2)
    meancolors = scatter_mean_2d(
        img.view(3, -1).mT,
        seg.view(-1)
    )
    bg_img = img if use_bg else None
    return bezier.to_svg(meancolors, stroke, bg_img, patch_size, 0, W, H).getvalue()

