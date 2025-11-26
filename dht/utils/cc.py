import torch
from torch import Tensor
from warnings import warn

def _dpcc_roots(labels: Tensor):
    device = labels.device
    changed = torch.ones(labels.size(0), dtype=torch.bool, device=device)
    
    while changed.any():
        idx = torch.nonzero(changed, as_tuple=False).squeeze()
        current_labels = labels[idx]
        new_labels = labels[current_labels]
        labels[idx] = new_labels
        changed[idx] = (new_labels != current_labels)
    return labels


def _dpcc_iterative(n:int, u:Tensor, v:Tensor, labels: Tensor, maxit:int = 64):
    m = len(u)
    it = 0

    while m > 0 and it <= maxit:
        l2h = (u < v).sum()
        h2l = m - l2h

        if l2h >= h2l:
            mask = u < v
        else:
            mask = u > v

        labels[u[mask]] = labels[v[mask]]

        # Compute roots
        _dpcc_roots(labels)

        # Compute new edges
        mask = labels[u] != labels[v]
        u = labels[u[mask]]
        v = labels[v[mask]]
        m = len(u)
        it += 1

    if it > maxit:
        msg = f'DPCC iteration limit - current iteration: {it} > max iterations: {maxit}.'
        warn(msg, RuntimeWarning)


def concom_full(src:Tensor, tgt:Tensor, n:int, maxit:int=64) -> Tensor:
    """Computes a tensor based on the provided inputs.

    Parameters
    ----------
    src : Tensor,
        The source tensor used for computation.
    tgt : Tensor
        The target tensor used for computation.
    n : int
        An integer parameter for the computation.
    maxit : int, optional
        The maximum number of iterations. Defaults to 64.

    Returns
    -------
    Tensor
        The resulting tensor after computation.
    """
    # assert src.ndim == 1
    # assert src.shape == tgt.shape
    # assert src.device == tgt.device
    
    # Init labels
    device = src.device
    labels = torch.arange(n, device=device)

    # Connected Components
    _dpcc_iterative(n, src, tgt, labels, maxit=maxit)

    # Return unique inverse
    return labels.unique(return_inverse=True)[1]


def concom_count(src:Tensor, tgt:Tensor, n:int, maxit:int=64) -> tuple[Tensor, int]:
    """Computes a tensor based on the provided inputs.

    Parameters
    ----------
    src : Tensor,
        The source tensor used for computation.
    tgt : Tensor
        The target tensor used for computation.
    n : int
        An integer parameter for the computation.
    maxit : int, optional
        The maximum number of iterations. Defaults to 64.
    
    Returns
    -------
    tuple[Tensor, int]
        The resulting tensor after computation and the count of unique elements.
    """
    # Init labels
    device = src.device
    labels = torch.arange(n, device=device)

    # Connected Components
    _dpcc_iterative(n, src, tgt, labels, maxit=maxit)

    # Return unique inverse
    unq, inv = labels.unique(return_inverse=True)
    return inv, len(unq)
