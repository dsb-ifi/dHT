import torch
from torch import Tensor
from typing import Optional

from .indexing import confusion_matrix

_valid_reductions = ('sum', 'prod', 'mean', 'amax', 'amin')

def scatter_reduce_2d(src:Tensor, idx:Tensor, red:str, nnz:Optional[int]=None) -> Tensor:
    '''Scatter reduction over dim 0 with 1/2d source and 1d index.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    red : str
        Reduction method.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    add_dim = False
    if src.ndim == 1:
        add_dim = True
        src = src.unsqueeze(-1)
    if idx.ndim == 1:
        idx = idx.unsqueeze(1).expand(*src.shape)
    if nnz is None:
        nnz = int(idx.max().item()) + 1
    out = src.new_empty(nnz, src.shape[1])
    out.scatter_reduce_(0, idx, src, red, include_self=False)
    if not add_dim: return out
    return out.squeeze(-1)


def scatter_mean_2d(src:Tensor, idx:Tensor, nnz:Optional[int]=None):
    '''Scatter mean reduction. Convenience function.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    return scatter_reduce_2d(src, idx, 'mean', nnz=nnz)


def scatter_max_2d(src:Tensor, idx:Tensor, nnz:Optional[int]=None):
    '''Scatter max reduction. Convenience function.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    return scatter_reduce_2d(src, idx, 'amax', nnz=nnz)


def scatter_min_2d(src:Tensor, idx:Tensor, nnz:Optional[int]=None):
    '''Scatter min reduction. Convenience function.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    return scatter_reduce_2d(src, idx, 'amin', nnz=nnz)


def scatter_sum_2d(src:Tensor, idx:Tensor, nnz:Optional[int]=None):
    '''Scatter sum reduction. Convenience function.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    return scatter_reduce_2d(src, idx, 'sum', nnz=nnz)


def scatter_prod_2d(src:Tensor, idx:Tensor, nnz:Optional[int]=None):
    '''Scatter prod reduction. Convenience function.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    return scatter_reduce_2d(src, idx, 'prod', nnz=nnz)


def scatter_var_2d(src:Tensor, idx:Tensor, nnz:Optional[int]=None) -> Tensor:
    '''Scatter variance reduction.

    NOTE: Runs two passes.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    mu = scatter_mean_2d(src, idx)
    diff = (src - mu[idx]).pow(2)
    return scatter_mean_2d(diff, idx, nnz=nnz)


def scatter_cov_2d(src:Tensor, idx:Tensor, nnz:Optional[int]=None) -> Tensor:
    '''Scatter covariance reduction.

    NOTE: Runs two passes.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    d = src.shape[-1]
    mu = scatter_mean_2d(src, idx, nnz=nnz)
    diff = (src - mu[idx])
    return scatter_mean_2d(
        (diff.unsqueeze(-1) @ diff.unsqueeze(-2)).view(-1,d**2), 
        idx,
        nnz=nnz
    )


def scatter_weighted_sum_2d(src:Tensor, weight:Tensor, idx:Tensor, nnz:Optional[int]=None) -> Tensor:
    '''Scatter weighted sum reduction.

    If the weights are a convex combination,  i.e. 
    for all indices `i` we have `sum(weight, index)[i] = 1`
    then this is equivalent to a weighted mean.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    weight : Tensor
        Weight tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    return scatter_sum_2d(weight * src, idx, nnz=nnz)


def scatter_range_2d(src:Tensor, idx:Tensor, nnz:Optional[int]=None) -> Tensor:
    '''Computes scattered range (max-min) with 2d source and 1d index.

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx : Tensor
        Index tensor.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    if idx.ndim == 1:
        idx = idx.unsqueeze(1).expand(*src.shape)
    if nnz is None:
        nnz = int(idx.max().item()) + 1
    mx = src.new_empty(nnz, src.shape[1])
    mn = src.new_empty(nnz, src.shape[1])
    mx.scatter_reduce_(0, idx, src, 'amax', include_self=False)
    mn.scatter_reduce_(0, idx, src, 'amin', include_self=False)
    return mx - mn


def scatter_bitpack_argmax_edges(
    val:Tensor, u:Tensor, v:Tensor, nnz:Optional[int]=None, 
    valbit:int=16,
):
    '''Compute the argmax indices for given values with bit packing.

    This function performs bit packing to combine values with indices,
    then uses scatter operations to compute the argmax values efficiently.
    The values are expected to be floats in the range [0,1].

    Parameters
    ----------
    val : torch.Tensor
        A tensor of values to be clipped and packed.
    u : torch.Tensor
        A tensor of indices associated with the first dimension.
    v : torch.Tensor
        A tensor of indices associated with the second dimension.
    nnz : int, optional
        Optional size of array, defaults to max(idx) + 1
    valbit : int, optional
        Number of bits to use for the values (default is 16).

    Returns
    -------
    torch.Tensor
        A tensor containing the argmax indices after bit packing.
    '''
    if nnz is None:
        nnz = max(int(u.max().item()) + 1, int(v.max().item()) + 1)
    idxbit = 63 - valbit
    shorts = val.clip(0,1).mul_(2**valbit - 1).long().bitwise_left_shift_(idxbit)
    packed_u = shorts | u
    packed_v = shorts | v
    packed_values = v.new_zeros(nnz, dtype=torch.long)
    packed_values.scatter_reduce_(0, v, packed_u, 'amax', include_self=False)
    packed_values.scatter_reduce_(0, u, packed_v, 'amax', include_self=True)  
    out = packed_values.bitwise_and(2**idxbit-1)
    
    # NOTE: If these are not correct, we risk getting invalid memory access
    assert (out.max().item() < nnz)
    assert (out.min().item() >= 0)
    return out


def scatter_bitpack_argmax_edges_and_values(
    val:Tensor, u:Tensor, v:Tensor, nnz:Optional[int]=None, 
    valbit:int=16
) -> tuple[Tensor, Tensor]:
    if nnz is None:
        nnz = max(int(u.max().item()) + 1, int(v.max().item()) + 1)
    idxbit = 63 - valbit
    shorts = val.clip(0,1).mul_(2**valbit - 1).long().bitwise_left_shift_(idxbit)
    packed_u = shorts | u
    packed_v = shorts | v
    packed_values = v.new_zeros(nnz, dtype=torch.long)
    packed_values.scatter_reduce_(0, v, packed_u, 'amax', include_self=False)
    packed_values.scatter_reduce_(0, u, packed_v, 'amax', include_self=True)  
    out = packed_values.bitwise_and(2**idxbit-1)
    
    # NOTE: If these are not correct, we risk getting invalid memory access
    assert (out.max().item() < nnz)
    assert (out.min().item() >= 0)

    values = packed_values >> idxbit
    return out, values / (2**valbit - 1)


def scatter_reduce_symmetric_2d(src:Tensor, idx1:Tensor, idx2:Tensor, red:str, nnz:Optional[int]=None):
    '''Symmetric scatter reduction for 2d data.

    Useful for reduction for contributions of edges. 

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx1 : Tensor
        Index tensor.
    idx2 : Tensor
        Index tensor.
    red : str
        Reduction type.
    nnz : int, optional
        Optional size of array, defaults to max(idx1, idx2) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    if nnz is None:
        nnz = max(int(idx1.max().item()), int(idx2.max().item())) + 1
    out = scatter_reduce_2d(src, idx1, red, nnz)
    out.scatter_reduce_(0, idx2, src, red, include_self=True)
    return out


def scatter_sum_symmetric_2d(src:Tensor, idx1:Tensor, idx2:Tensor, nnz:Optional[int]=None):
    '''Symmetric scatter sum reduction for 2d data.

    Useful for summing contributions of edges. 

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx1 : Tensor
        Index tensor.
    idx2 : Tensor
        Index tensor.
    red : str
        Reduction type.
    nnz : int, optional
        Optional size of array, defaults to max(idx1, idx2) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    return scatter_reduce_symmetric_2d(src, idx1, idx2, 'sum', nnz)


def scatter_mean_symmetric_2d(src:Tensor, idx1:Tensor, idx2:Tensor, nnz:Optional[int]=None):
    '''Symmetric scatter mean reduction for 2d data.

    Useful for averaging contributions of edges. 

    Parameters
    ----------
    src : Tensor 
        Source tensor.
    idx1 : Tensor
        Index tensor.
    idx2 : Tensor
        Index tensor.
    red : str
        Reduction type.
    nnz : int, optional
        Optional size of array, defaults to max(idx1, idx2) + 1

    Returns
    -------
    Tensor
        Output tensor.
    '''
    return scatter_reduce_symmetric_2d(src, idx1, idx2, 'mean', nnz)


def _scatter_confusion_matrix(pred:Tensor, target:Tensor, n_classes:Optional[int]=None) -> Tensor:
    '''Computes a confusion matrix using scatter operations.

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
    if n_classes is None:
        n_classes = max(
            int(target.max().item()) + 1,
            int(pred.max().item()) + 1
        )
    
    flat = pred.view(-1) * n_classes + target.view(-1)
    return scatter_sum_2d(
        torch.ones_like(flat, dtype=torch.float32), 
        flat, 
        nnz=n_classes**2
    ).view(n_classes, n_classes)

# TODO: Quick fix, debug scatter version
scatter_confusion_matrix = confusion_matrix

def _jaccard_from_confusion_matrix(
    cm:Tensor, weights:Optional[Tensor]=None, 
    ignore_index:Optional[int]=-1, eps:float=1e-7
) -> Tensor:
    n_classes = cm.shape[0]
    num = torch.diag(cm)
    fnfp = cm.sum(0) + cm.sum(1)
    den = fnfp - num + eps
    jac = num / den
    if weights is None:
        weights = torch.ones_like(jac)
        if ignore_index is not None and 0 <= ignore_index:
            if ignore_index >= n_classes:
                msg1 = f'Ignore index higher than inferred number of classes: '
                msg2 = '{ignore_index} >= {n_classes}'
                raise ValueError(msg1 + msg2)
            weights[ignore_index] = 0
        weights[fnfp == 0] = 0
    return (weights * jac / weights.sum()).sum()



def scatter_jaccard_index(
    pred:Tensor, target:Tensor, n_classes:Optional[int]=None, 
    weights:Optional[Tensor]=None, ignore_index:Optional[int]=-1, 
    eps:float=1e-8
) -> Tensor:
    '''Computes multiclass Jaccard index using scatter operations.

    Parameters
    ----------
    pred : Tensor
        Predicted labels, corresponds to rows in conf. matrix.
    target : Tensor
        Predicted targets, corresponds to columns in conf. matrix.
    n_classes : int, optional
        Number of classes in the prediction.
    weights : Tensor, optional
        Class weight tensor.
    ignore_index : int
        Index to ignore, defaults to zero. 
        If negative or None, no index is ignored.
    eps : float
        Epsilon for numerically robust division.

    Returns
    -------
    Tensor
        Jaccard Index / mIoU of observations.
    '''
    cm = confusion_matrix(pred, target, n_classes)
    return _jaccard_from_confusion_matrix(
        cm, weights, ignore_index, eps
    )   
