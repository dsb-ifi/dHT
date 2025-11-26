import torch

from torch import Tensor
from typing import Optional, Callable


def _gaussian(fV: Tensor, E: Tensor) -> Tensor:
    return fV[E].diff(1,0)[0].pow(2).sum(-1).neg_().exp_()

_simdict = dict(gaussian=_gaussian)

def get_similarity(similarity:str) -> Callable[[Tensor,Tensor], Tensor]:
    return _simdict.get(similarity, _gaussian)

def compute_edge_features(
    fV: Tensor, E: Tensor, mV: Tensor, similarity: str = 'gaussian',
    bb: Optional[Tensor]=None, cmp: float=0.0, center: float=0.5,
    projs2: Optional[Tensor]=None
) -> Tensor:
    '''Computes edge features based on vertex features and the given energy function. 
    
    NOTE: Optionally regularizes compactness using bounding box constraints.

    Parameters
    ----------
    fV : Tensor
        Float tensor containing features for each vertex, shape [nV, C].
    E : Tensor
        Long tensor specifying edges between vertices, shape [2, k].
    mV : Tensor
        Long tensor, number of aggregated vertices for each current vertex, shape [nV].
    similarity : str
        String indicating energy function to apply between connected vertices.
    bb : Tensor, optional
        Tensor containing bounding box coordinates for each vertex, shape [4, nV].
    cmp : float, optional
        Compactness factor to regularize based on bounding box circumference.
    center : float, optional
        Initial value to populate the edge features tensor.

    Returns
    -------
    Tensor
        Computed edge features, shape [k].
    '''
    fE = fV.new_full((E.shape[1],), center)
    nloop = torch.ne(*E)        
    loop = ~nloop
    fE[nloop] = get_similarity(similarity)(fV, E[:,nloop])
    if projs2 is not None:
        fE[loop] = projs2.sigmoid()[E[0,loop], 0]

    if cmp > 0 and bb is not None:
        nE = E[:,nloop]
        l0 = E[0,loop]
        mVf = mV.to(fE.dtype)
        ymin, xmin, ymax, xmax = bb.to(fE.dtype)
        
        fC = torch.zeros_like(fE)
        fC[loop] = 4*cmp*mVf[l0] / (
            2 + ymax[l0] - ymin[l0] + xmax[l0] - xmin[l0]
        ).pow(2)

        fC[nloop] = 4*cmp*mVf[nE].sum(0) / (2 +
            ymax[nE].max(0)[0] - ymin[nE].min(0)[0] +
            xmax[nE].max(0)[0] - xmin[nE].min(0)[0]
        ).pow(2)
        
        fE.mul_(1 - cmp).add_(fC)
        
    return fE