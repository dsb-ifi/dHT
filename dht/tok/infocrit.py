import torch

from torch import Tensor

def gaussian_2nll(s2:Tensor, mV:Tensor, iota:float=1) -> Tensor:
    '''Computes the Gaussian negative log-likelihood for variance and vertex counts.

    Parameters
    ----------
    s2 : Tensor
        Tensor of variance for each vertex, shape [nV, C].
    mV : Tensor
        Tensor of counts of aggregated vertices, shape [nV].
    iota : float, optional
        Small constant to ensure numerical stability.

    Returns
    -------
    Tensor
        The Gaussian negative log-likelihood, adjusted for the vertex counts.
    '''
    return mV * (2*torch.pi*s2).prod(-1).add(iota).log()

def _df(mV:Tensor) -> Tensor: return 1/(2*mV - 2*mV.sqrt())

def _aic_fn(mV:Tensor) -> Tensor: return 2 * _df(mV)

def _bic_fn(mV:Tensor) -> Tensor: return mV.log() * _df(mV)

def _cic_fn(mV:Tensor) -> Tensor: return mV.log().add(1) * _df(mV)

def _aicc_fn(mV:Tensor) -> Tensor: k = _df(mV); return 2*k + 2*(k**2 + k) / (mV - k - 1)

_infodict = dict(aic=_aic_fn, aicc=_aicc_fn, bic=_bic_fn, cic=_cic_fn)


def infocrit(
    s2:Tensor, mV:Tensor, H:int, W:int, 
    mode:str='aic', iota:float=1, eps:float=1e-8
) -> Tensor:
    '''Computes information criteria for model selection among segmented images.

    Parameters
    ----------
    s2 : Tensor
        Tensor of variance for each vertex, shape [nV, C].
    mV : Tensor
        Tensor of counts of aggregated vertices, shape [nV].
    H : int
        Height of the image.
    W : int
        Width of the image.
    mode : str, optional
        Specifies which information criterion to use ('aic', 'bic', 'aicc').
    iota : float, optional
        Small constant to ensure numerical stability in log calculations.
    eps : float, optional
        Small constant to avoid division by zero.

    Returns
    -------
    Tensor
        The calculated negative information criterion value for model comparison.
    '''
    nc = (H*W)
    nll = gaussian_2nll(s2, mV/nc, iota)
    fn = _infodict.get(mode, _aic_fn)
    k = fn(mV + eps)
    return (nll + k).neg()
