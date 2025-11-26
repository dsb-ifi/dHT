import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Union, Sequence, Optional

def unravel_index(
    idx:Tensor, shape:Union[Sequence[int], Tensor]
) -> Tensor:
    '''Converts a tensor of flat indices into a tensor of coordinate vectors.

    Parameters
    ----------
    idx : Tensor 
        Indices to unravel.
    shape : tuple[int] 
        Shape of tensor.

    Returns
    -------
    Tensor
        Tensor (long) of unraveled indices.
    '''
    try:
        shape = idx.new_tensor(torch.Size(shape))[:,None] # type: ignore
    except Exception:
        pass
    shape = F.pad(shape, (0,0,0,1), value=1)                  # type: ignore
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()        # type: ignore
    return torch.div(idx[None], coefs, rounding_mode='trunc') % shape[:-1]


def fast_uidx_1d(src:Tensor) -> Tensor:
    '''Pretty fast unique index calculation for 1d tensors.

    Parameters
    ----------
    src : Tensor
        Tensor to compute unique indices for.

    Returns
    -------
    Tensor
        Tensor (long) of indices.
    '''
    # assert ar.ndim == 1, f'Need dim of 1, got: {ar.ndim}!'
    perm = src.argsort()
    aux = src[perm]
    mask = src.new_zeros(aux.shape[0], dtype=torch.bool)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    return perm[mask]


def fast_uidx_long2d(src:Tensor) -> Tensor:
    '''Pretty fast unique index calculation for 2d long tensors (row wise).

    Parameters
    ----------
    src : Tensor
        Tensor to compute unique indices for.

    Returns
    -------
    Tensor
        Tensor (long) of indices.
    '''
    # assert ar.ndim == 2, f'Need dim of 2, got: {ar.ndim}!'
    m = src.max() + 1
    r, c = src
    cons = r*m + c
    return fast_uidx_1d(cons)


def fast_invuidx_1d(src: Tensor) -> tuple[Tensor, Tensor]:
    '''Fast unique index and inverse calculation for 1d tensors.

    Parameters
    ----------
    src : Tensor
        The 1-dimensional tensor for which to compute unique indices.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing:
        - A tensor of indices that can be used to reconstruct the original tensor from the unique values.
        - A tensor of inverse indices that maps the original tensor to these unique values.
    '''
    assert src.ndim == 1, f'Need dim of 1, got: {src.ndim}!'
    perm = src.argsort()
    aux = src[perm]
    mask = src.new_zeros(aux.shape[0], dtype=torch.bool)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    
    imask = mask.cumsum(-1) - 1
    inv_idx = torch.empty_like(mask, dtype=src.dtype)
    inv_idx[perm] = imask
    return perm[mask], inv_idx


def fast_invuidx_long2d(src: Tensor) -> tuple[Tensor, Tensor]:
    '''Fast unique index and inverse calculation for 3d tensors.

    Parameters
    ----------
    src : Tensor
        The tensor to compute unique indices for, expected to be 3-dimensional.

    Returns
    -------
    Tensor
        A tensor of indices and inverse mappings, returned as a tuple.
    '''
    assert src.ndim == 2, f'Need dim of 2, got: {src.ndim}!'
    assert src.shape[0] == 2, f'Need dim 0 shape of 2, got: {src.shape[0]}!'
    a, b = src
    mb = b.max() + 1
    cons = mb*a + b
    return fast_invuidx_1d(cons)


def fast_invuidx_long3d(src: Tensor) -> tuple[Tensor, Tensor]:
    '''Fast unique index and inverse calculation for 3d tensors.

    Parameters
    ----------
    src : Tensor
        The tensor to compute unique indices for, expected to be 3-dimensional.

    Returns
    -------
    Tensor
        A tensor of indices and inverse mappings, returned as a tuple.
    '''
    assert src.ndim == 2, f'Need dim of 2, got: {src.ndim}!'
    assert src.shape[0] == 3, f'Need dim 0 shape of 3, got: {src.shape[0]}!'
    a, b, c = src
    mb, mc = b.max() + 1, c.max() + 1
    cons = mb*mc*a + mc*b + c
    return fast_invuidx_1d(cons)


def lexsort_old(src:Tensor, dim:int=-1) -> Tensor:
    '''Lexicographical sort of multidimensional tensor.

    Parameters
    ----------
    src : Tensor
        Input tensor.
    dim : int
        Dimension to sort over, defaults to -1.

    Returns
    -------
    Tensor
        Sorting indices for multidimensional tensor.
    '''
    if src.ndim < 2:
        raise ValueError(f"src must be at least 2 dimensional, got {src.ndim=}.")
    if len(src) == 0:
        raise ValueError(f"Must have at least 1 key, but {len(src)=}.")
    
    idx = src[0].argsort(dim=dim, stable=True)
    for k in src[1:]:
        idx = idx.gather(dim, k.gather(dim, idx).argsort(dim=dim, stable=True))
    
    return idx


def lexsort(*tensors:Tensor) -> Tensor:
    '''Lexicographical sort of multidimensional tensor.

    Parameters
    ----------
    src : Tensor
        Input tensor.
    dim : int
        Dimension to sort over, defaults to -1.

    Returns
    -------
    Tensor
        Sorting indices for multidimensional tensor.
    '''
    numel = tensors[0].numel()
    assert all([t.ndim == 1 for t in tensors])
    assert all([t.numel() == numel for t in tensors[1:]])
    idx = tensors[0].argsort(dim=0, stable=True)
    for k in tensors[1:]:
        idx = idx.gather(0, k.gather(0, idx).argsort(dim=0, stable=True))
    return idx


def confusion_matrix(pred:Tensor, target:Tensor, n_classes:Optional[int]) -> Tensor:
    '''Returns a confusion matrix for targets and labels.

    Parameters
    ----------
    pred : Tensor
        Predicted labels, corresponds to rows in conf. matrix.
    target : Tensor
        Predicted targets, corresponds to columns in conf. matrix.
    n_classes : int, optional
        Number of classes in the prediction.
    
    Returns
    -------
    Tensor
        Inferred confusion matrix of shape (n_classes, n_classes) where
        rows correspond to predictions and columns correspond to targets.
    '''
    inferred_classes = False
    if n_classes is None:
        inferred_classes = True
        n_classes = max(
            int(target.max().item()) + 1,
            int(pred.max().item()) + 1
        )

    bc = (pred.view(-1) * n_classes + target.view(-1)).bincount()
    if inferred_classes:
        return bc.view(n_classes, n_classes)
    return F.pad(bc, (0,n_classes**2 - len(bc))).view(n_classes, n_classes)
    