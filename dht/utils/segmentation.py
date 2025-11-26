import torch

from torch import Tensor
from torch.nn import functional as F
from typing import Optional
from .cc import concom_full as connected_components
from .indexing import fast_uidx_long2d, unravel_index

def get_hierarchy_sublevel(
    hierograph:list[Tensor], level1:int, level2:Optional[int]=None
) -> Tensor:
    '''Retrieves superpixel mapping between level1 to level2.
    
    Parameters
    ----------
    hierograph : List[Tensor]
        List of mapping tensors.
    level1 : int
        Level to compute mapping from.
    level2 : int
        Level to compute mapping to.

    Returns
    -------
        Tensor: Mapping of indices from level1 to level2.
    '''
    nlvl = len(hierograph)
    if level2 is None:
        level2 = nlvl - 1

    if not (0 <= level1 < nlvl and 0 <= level2 < nlvl):
        raise ValueError("Invalid hierarchy levels")

    if level1 == level2:
        return hierograph[level1]
    
    min_level, max_level = min(level1, level2), max(level1, level2)

    segmentation = hierograph[min_level]
    for i in range(min_level + 1, max_level + 1):
        segmentation = hierograph[i][segmentation]

    return segmentation


def get_seg_edges(seg:Tensor, B:int, H:int, W:int, unique:bool=False, return_edges:bool=False) -> Tensor:
    '''Computes edges between partitions, given a segmentation.

    Parameters
    ----------
    seg : Tensor
        Flat segmentation tensor.
    B : int
        Batch size.
    H : int
        Height of image.
    W : int
        Width of image.
    unique : bool
        Flag to ensure unique edges.

    Returns
    -------
    Tensor: Computed edge tensors.
    '''
    lr = seg.view(B, H, W).unfold(-1, 2, 1).reshape(-1, 2).mT
    ud = seg.view(B, H, W).unfold(-2, 2, 1).reshape(-1, 2).mT
    edges = torch.cat([lr, ud], -1)
    if unique:
        uidx = fast_uidx_long2d(edges)
        return edges[:,uidx]
    return edges


def relabel_concom(seg:Tensor, B:int, H:int, W:int) -> Tensor:
    '''Relabels segmentation with connected components.

    Parameters
    ----------
    seg : Tensor
        Flat segmentation tensor.
    B : int
        Batch size.
    H : int
        Height of image.
    W : int
        Width of image.

    Returns
    -------
    Tensor: Relabeled segmentation.
    '''
    device = seg.device
    E = get_seg_edges(torch.arange(B*H*W, device=device), B, H, W)
    u, v = E[:,seg[E[0]] == seg[E[1]]]
    return connected_components(u, v, B*H*W)


def init_image_graph(img:Tensor) -> tuple[Tensor, Tensor, Tensor, int, Tensor]:
    '''Initializes a first order pixel graph from an image.

    Parameters
    ----------
    img (Tensor):
        Input image.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, int, Tensor]: Set of graph tensors.
    '''
    B,C,H,W = img.shape
    nnz = B*H*W
    lab = torch.arange(nnz, device=img.device)
    sizes = torch.ones_like(lab)
    coords = unravel_index(lab, (B,H,W))

    # Essentially get_seg_edges, optionally rewrite?
    lr = lab.view(B, H, W).unfold(-1, 2, 1).reshape(-1, 2).mT
    ud = lab.view(B, H, W).unfold(-2, 2, 1).reshape(-1, 2).mT
    edges = torch.cat([lr, ud], -1)

    return lab, edges, sizes, nnz, coords


def get_batch_index(
    seg:Tensor, b:Tensor, B:int, return_token_index:bool=False
) -> Tensor:
    '''Returns batch indices per token for a segmentation.

    Parameters
    ----------
    seg : Tensor
        The segmentation tensor.
    b : Tensor
        Batch indices per pixel.
    B : int
        Batch size

    Returns
    -------
    Tensor
        Batch indices per token.
    '''
    b_idx = seg.mul(B).add_(b).unique() % B
    if not return_token_index:
        return b_idx
    bc = b_idx.bincount()
    cs = bc.cumsum(-1)
    st = cs - bc
    s_idx = (
        torch.arange(len(b_idx), device=b_idx.device) - 
        st.repeat_interleave(bc)
    )
    return torch.stack([b_idx, s_idx], 0)


def _get_token_mask(
    seg:Tensor, b:Optional[Tensor], B:int, 
    fill:Optional[Tensor]=None, b_idx:Optional[Tensor]=None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if b_idx is None:
        if b is None:
            raise NotImplementedError('Token mask requires either b or b_idx.')
        b_idx = get_batch_index(seg, b, B)
    bc = b_idx.bincount()
    cs = bc.cumsum(-1)
    st = cs - bc
    s_idx = (
        torch.arange(len(b_idx), device=b_idx.device) - 
        st.repeat_interleave(bc)
    )
    maxdim = int(bc.max().item()) + 1
    if fill is not None:
        mask = fill.new_zeros(B, maxdim)
        mask[b_idx, s_idx] = fill
        return mask, b_idx, bc, st
    
    mask = b_idx.new_zeros(B, maxdim, dtype=torch.bool)
    mask[b_idx, s_idx] = True
    return mask, b_idx, bc, st
    
    
def get_token_mask(
    seg:Tensor, b:Tensor, B:int, fill:Optional[Tensor]=None    
) -> Tensor:
    '''Returns a token mask for a segmentation.

    Yields a mask of shape [B, max(|V|)]. Defaults to filling the mask
    with boolean values where True indicates that the batch has a token
    in the position indicated by the mask.

    Parameters
    ----------
    seg : Tensor
        The segmentation tensor.
    b : Tensor
        Batch indices per pixel.
    B : int
        Batch size
    fill : Tensor, optional
        A tensor of values to fill in the token mask. Defaults to True.

    Returns
    -------
    Tensor
        Token mask.
    '''
    return _get_token_mask(seg, b, B, fill)[0]


def uniform_square_partitions(
    b:int, h:int, w:int, p:int, 
    device:torch.device=torch.device('cpu')
) -> torch.Tensor:
    '''Generates uniform square partitions of dimension BxHxW.
    
    Args:
        b (int): Batch size.
        h (int): Raster height.
        w (int): Raster width.
        p (int): Partition size.
        device (torch.device): Output device.
    
    Returns:
        torch.Tensor: Square partition.
    '''
    ceil_h, ceil_w = -(-h//p), -(-w//p)
    partition_ids = torch.arange(b * ceil_h * ceil_w, device=device)    
    return (
        partition_ids.view(b, ceil_h, ceil_w)
            .repeat_interleave(p, dim=-2)
            .repeat_interleave(p, dim=-1)
            [...,:h,:w]
    )


def binary_edge_mask(seg:Tensor, pad_border:bool=False, fill_border:bool=False) -> Tensor:
    '''Computes a binary edge mask for a segmentation tensor.

    Parameters
    ----------
    seg : Tensor
        The segmentation tensor.
    pad_border : bool, optional
        Whether to pad the border with True values. Defaults to False.
    fill_border : bool, optional
        Whether to fill the border with True values. Defaults to False.

    Returns
    -------
    Tensor
        Binary edge mask.
    '''
    borders = (
        seg.diff(n=1, dim=-1, append =seg[...,:,-1,None]).ne(0) | 
        seg.diff(n=1, dim=-1, prepend=seg[...,:, 0,None]).ne(0) | 
        seg.diff(n=1, dim=-2, append =seg[...,None,-1,:]).ne(0) | 
        seg.diff(n=1, dim=-2, prepend=seg[...,None, 0,:]).ne(0)
    )
    
    if pad_border:
        return F.pad(borders, (1,1,1,1), value=True)

    if fill_border:
        return F.pad(borders[...,1:-1,1:-1], (1,1,1,1), value=True)

    return borders

