import torch
import torch.nn as nn

from torch import Tensor
from typing import NamedTuple, Optional, Callable
from .infocrit import infocrit
from .similarity import compute_edge_features, get_similarity
from ..utils.scatter import scatter_sum_2d, scatter_mean_2d, scatter_bitpack_argmax_edges
from ..utils.indexing import fast_invuidx_1d, fast_uidx_1d
from ..utils.cc import concom_count as connected_components
from ..utils.segmentation import relabel_concom, get_seg_edges, _get_token_mask
from ..utils.ras2vec import ras2svg

class GridDimensions(NamedTuple):
    '''NamedTuple to hold grid dimensions.

    Attributes
    ----------
    byx : Tensor
        Grid coordinate tensor, (batch, y, x).
    B : int
        Batch size dimension.
    C : int
        Channel size dimension.
    H : int
        Height dimension.
    W : int
        Width dimension.
    '''
    byx: Tensor
    B: int
    C: int
    H: int
    W: int

class TokenizerState(NamedTuple):
    '''NamedTuple to hold the state of a tokenizer with multiple tensor attributes.

    Attributes
    ----------
    fV : Tensor
        Float tensor, current vertex features.
    s2 : Tensor
        Float tensor, current diagonal variance.
    E : Tensor
        Long tensor, edges of image graph.
    mV : Tensor
        Long tensor, number of aggregated vertices for each current vertex.
    info : Tensor
        Float tensor, estimated information criteria per vertex.
    curseg : Tensor
        Long tensor, the current segment indices.
    optseg : Tensor
        Long tensor, optimal seg. regions given information criteria.
    optinfo : Tensor
        Float tensor, containing optimal information criteria per vertex.
    optfV : Tensor
        Float tensor, optional vertex features given information criteria.
    opts2 : Tensor
        Float tensor, optional vertex variances given information criteria.
    nV : int
        Number of vertices.
    optnV : int
        Number of regions in `optseg`.
    grid : GridDimensions
        NamedTuple with coordinates and shape info.
    bb : Tensor, optional
        Bounding box coordinates for each region.
    wV : Tensor, optional.
        Optional argmax edge potentials for parameter updates.
    '''
    fV: Tensor
    s2: Tensor
    E: Tensor
    mV: Tensor
    info: Tensor
    curseg: Tensor
    optseg: Tensor
    optinfo: Tensor
    optfV: Tensor
    opts2: Tensor
    nV: int
    optnV: int
    grid: GridDimensions
    bb: Optional[Tensor] = None
    wV: Optional[Tensor] = None


class TokenizerFeatures(NamedTuple):
    '''NamedTuple holding tokenizer features and segmentation.

    Attributes
    ----------
    fV : Tensor
        Float tensor, final vertex features.
    s2 : Tensor
        Float tensor, final diagonal variance.
    seg : Tensor
        Segmentation tensor, mapping to original pixel locations.
    nV : int
        Number of vertices, regions.
    bb : Tensor
        Bounding box coordinates of regions.
    grid : GridDimensions
        Image coordinates
    '''
    fV: Tensor
    seg: Tensor
    nV: int
    bb: Tensor
    grid: GridDimensions

    def to_svg(
        self, img:Tensor, th0:float=0., th1:float=0.1, th2:float=0.1, 
        use_bg:bool=True, image_index:int=0, stroke:float=1.0, patch_size:int=32,
        device:torch.device=torch.device('cpu')
    ) -> str:
        B, H, W = self.grid.B, self.grid.H, self.grid.W
        seg = self.seg.view(B, H, W)
        return ras2svg(
            seg.to(device), img.to(device), th0=th0, th1=th1, th2=th2, 
            use_bg=use_bg, image_index=image_index, 
            stroke=stroke, patch_size=patch_size
        )

def update_edges(cc: Tensor, E: Tensor, nV: int) -> Tensor:
    '''
    Updates and filters edges based on connectivity changes.

    Parameters
    ----------
    cc : Tensor
        Tensor of connected component labels, shape [nV].
    E : Tensor
        Long tensor specifying edges of the image graph, shape [2, k].
    nV : int
        Number of vertices.

    Returns
    -------
    Tensor
        Updated edges after filtering duplicates and reordering, shape [2, l] where l <= k.
    '''
    Ep = cc[E] 
    Eu = Ep[0]*nV + Ep[1]
    p = Eu.argsort()
    s = Eu[p]
    m = Eu.new_zeros(E.shape[1], dtype=torch.bool)
    m[:1] = True
    m[1:] = s[1:] != s[:-1]
    return Ep[:,p[m]]


def update_bbox(cc: Tensor, bb: Optional[Tensor], nV: int) -> Optional[Tensor]:
    '''
    Updates bounding boxes based on new connectivity labels.

    Parameters
    ----------
    cc : Tensor
        Tensor of connected component labels, shape [nV].
    bb : Tensor, optional
        Tensor containing original bounding box coordinates, shape [4, nV].
    nV : int
        Number of vertices.

    Returns
    -------
    Optional[Tensor]
        Updated bounding boxes if bb is not None, otherwise returns None, shape [4, nV].
    '''
    if bb is None: return bb
    bbp = bb.new_zeros(4, nV)
    bbp[0].scatter_reduce_(0, cc, bb[0], 'amin', include_self=False)
    bbp[1].scatter_reduce_(0, cc, bb[1], 'amin', include_self=False)
    bbp[2].scatter_reduce_(0, cc, bb[2], 'amax', include_self=False)
    bbp[3].scatter_reduce_(0, cc, bb[3], 'amax', include_self=False)
    return bbp


def update_params(
    cc: Tensor, fV: Tensor, s2: Tensor, mV: Tensor, 
    nV: int, wV: Optional[Tensor]=None
) -> tuple[Tensor, Tensor, Tensor]:
    '''
    Updates feature vectors, variances, and vertex counts based on new connectivity labels.

    NOTE: When wV is not supplied, the parameter update uses the relative sizes (mV) to weight the 
          updates of the mean and variances, which results in theoretically correct updates to the 
          distribution parameters. When wV is supplied, we instead use the argmax edge potentials
          as weights, thereby weighting the contributions by their similarity scores. This provides
          an additional source of gradient information to the updates while reducing the impact of
          potential outliers / spurious aggregations in the parameters. 

    Parameters
    ----------
    cc : Tensor
        Tensor of connected component labels, shape [nV].
    fV : Tensor
        Tensor of vertex features, shape [nV, C].
    s2 : Tensor
        Tensor of variances for each vertex, shape [nV, C].
    mV : Tensor
        Tensor of counts of aggregated vertices, shape [nV].
    nV : int
        Number of vertices.
    wV : Tensor, optional
        Optional supplied vertex aggregation weights, typically from argmax edge energies. 
        If `None`, uses relative sizes instead.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Updated feature vectors, variances, and vertex counts, shapes [nV, C], [nV, C], and [nV], respectively.
    '''
    mVp = scatter_sum_2d(mV, cc, nV)
    if wV is not None:
        num_w = wV.view(-1,1).sqrt() * mV.view(-1,1)
        den_w = scatter_sum_2d(num_w, cc, nV)
    else:
        num_w = mV.view(-1,1)
        den_w = mVp.view(-1,1)
    fVp = scatter_sum_2d(num_w*fV, cc, nV) / den_w
    s2p = scatter_sum_2d(num_w*(s2 + (fV - fVp[cc]).pow(2)), cc, nV) / den_w
    return fVp, s2p, mVp


def select_optimal_features(
    maxmask: Tensor, uidx: Tensor, cinfo: Tensor, minfo: Tensor,
    fV: Tensor, optfV: Tensor, s2: Tensor, opts2: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    '''
    Selects optimal features based on information criteria comparison.

    Parameters
    ----------
    maxmask : Tensor
        Boolean mask indicating where current info is better than optimal, shape [N].
    uidx : Tensor
        Unique indices for selection, shape [N].
    cinfo : Tensor
        Current information criteria values, shape [M].
    minfo : Tensor
        Maximum information criteria values observed so far, shape [M].
    fV : Tensor
        Current vertex features, shape [M, D].
    optfV : Tensor
        Optimal vertex features, shape [M, D].
    s2 : Tensor
        Current vertex variances, shape [M, D].
    opts2 : Tensor
        Optimal vertex variances, shape [M, D].

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Updated optinfo, optfV, and opts2 tensors.
    '''
    D = fV.shape[-1]
    maxmask_at_uidx = maxmask[uidx]
    optinfo = torch.where(maxmask_at_uidx, cinfo[uidx], minfo[uidx])
    optfV = torch.where(maxmask_at_uidx.view(-1,1).expand(-1,D), fV, optfV)
    opts2 = torch.where(maxmask_at_uidx.view(-1,1).expand(-1,D), s2, opts2)
    return optinfo, optfV, opts2


def update_info(
    curseg: Tensor, info: Tensor, optseg: Tensor, optinfo: Tensor,
    fV: Tensor, optfV: Tensor, s2: Tensor, opts2: Tensor,
    optnV: int, it: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, int, bool]:
    '''
    Updates segmentation information using provided criteria and iterative adjustments,
    to handle the accumulation and comparison of segment-related information.

    Parameters
    ----------
    curseg : Tensor
        Tensor of current segment indices, shape [BHW].
    info : Tensor
        Tensor of estimated information criteria per vertex, shape [BHW].
    optseg : Tensor
        Tensor of maximal segment indices observed, shape [BHW].
    optinfo : Tensor
        Tensor containing maximum values for information criteria per vertex, shape [BHW].
    fV : Tensor
        Tensor of current vertex features.
    optfV : Tensor
        Tensor of vertex features corresponding to maximum information criteria.
    s2 : Tensor
        Tensor of current vertex variances.
    opts2 : Tensor
        Tensor of vertex variances corresponding to maximum information criteria.
    optnV : int
        Maximum number of vertices (or max index of `optseg`).
    it : int
        Current iteration number.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, int]
        Updated `optinfo`, `optseg`, `optfV`, `opts2`, `optnV` plus a stop criterion.

    Notes
    -----
    The function operates differently for the initial iteration (it == 0) by initializing values
    and setting up the initial state of the maximum information tracking. For subsequent iterations,
    it adjusts the segments and information based on the updated criteria, continuously refining
    the segmentation process.
    '''
    if it == 0:
        cinfo = info[curseg]
        minfo = optinfo[optseg]
        maxmask = minfo <= cinfo
        uidx, new_optseg = fast_invuidx_1d(torch.where(maxmask, curseg + optnV, optseg))
        csi, msi = curseg[uidx], optseg[uidx]
        fV, optfV = fV[csi], optfV[msi]
        s2, opts2 = s2[csi], opts2[msi]
        optnV = int(new_optseg.max().item()) + 1
        optinfo, optfV, opts2 = select_optimal_features(
            maxmask, uidx, cinfo, minfo, fV, optfV, s2, opts2
        )
        return optinfo, new_optseg, optfV, opts2, optnV, False
        
    unq, inv = (optnV * curseg + optseg).unique(return_inverse=True)
    cunq, munq = unq // optnV, unq % optnV
    cinfo = info[cunq]
    minfo = optinfo[munq]
    maxmask = minfo <= cinfo
    stopcrit = not maxmask.any().item()
    uidx, invs = fast_invuidx_1d(torch.where(maxmask, cunq + optnV, munq))
    csi, msi = cunq[uidx], munq[uidx]
    fV, optfV = fV[csi], optfV[msi]
    s2, opts2 = s2[csi], opts2[msi]
    optseg = invs[inv]
    optnV = len(uidx)
    optinfo, optfV, opts2 = select_optimal_features(
        maxmask, uidx, cinfo, minfo, fV, optfV, s2, opts2
    )
    return optinfo, optseg, optfV, opts2, optnV, stopcrit


def update_tokenizer_state(
    state:TokenizerState, it:int,
    similarity:str='gaussian', criterion:str='aic', 
    cmp:float=0, center:float=0.5, iota:float=1, eps:float=1e-8,
    varproj:Optional[Callable[[Tensor], Tensor]]=None
) -> tuple[TokenizerState, bool]:
    '''Updates the tokenizer state.

    Parameters
    ----------
    state : TokenizerState
        The current state of the tokenizer, containing all relevant tensors and configuration values.
    it : int
        Current iteration number.
    similarity : str, optional
        Specifies the method to calculate edge similarities, default is 'gaussian'.
    criterion : str, optional
        Specifies the information criterion used to compute info values, default is 'aic'.
    cmp : float, optional
        Comparison factor for competitive interactions during edge feature computation.
    center : float, optional
        Initial value to populate the edge features tensor.
    iota : float, optional
        Small constant to ensure numerical stability in log calculations.
    eps : float, optional
        Small constant to avoid division by zero in calculations.

    Returns
    -------
   tuple[TokenizerState, bool]
        Updated state of the tokenizer and flag for early stopping.

    Notes
    -----
    This function orchestrates an update of the segmentation state by re-evaluating and recomputing
    the connections between vertices based on edge features and segmentation criteria, and adjusting
    segmentation based on new connectivity and bounding box recalculations.
    '''
    projs2 = None
    if varproj is not None:
        projs2 = varproj(state.s2)
    fE = compute_edge_features(
        state.fV, state.E, state.mV, similarity, state.bb, cmp, center, projs2
    )
    amx = scatter_bitpack_argmax_edges(fE, state.E[0], state.E[1], state.nV)
    cc, nV = connected_components(torch.arange(state.nV, device=amx.device), amx, state.nV)
    wV = fE[amx]
    curseg = cc[state.curseg]
    E = update_edges(cc, state.E, nV)
    bb = update_bbox(cc, state.bb, nV)
    fV, s2, mV = update_params(cc, state.fV, state.s2, state.mV, nV, wV)
    info = infocrit(s2, mV, state.grid.H, state.grid.W, criterion, iota, eps)
    optinfo, optseg, optfV, opts2, optnV, stopcrit = update_info(
        curseg, info, state.optseg, state.optinfo, 
        fV, state.optfV, s2, state.opts2, 
        state.optnV, it
    )
    new_state = TokenizerState(
        fV, s2, E, mV, info, curseg, optseg, optinfo, optfV, opts2, 
        nV, optnV, state.grid, bb, wV
    )
    return new_state, stopcrit


def finalize_tokenization(
    state:TokenizerState, origE: Tensor,
    criterion:str='aic', iota:float=1, eps:float=1e-8,
    proj:nn.Module = nn.Identity(),
    pretrain:bool = False
) -> TokenizerFeatures:
    '''Finalizes tokenization process.

    Parameters
    ----------
    state : TokenizerState
        The current state of the tokenizer, containing all relevant tensors and configuration values.
    origE : Tensor
        Tensor of original grid edges, used to finalize segmentation.
    criterion : str, optional
        Specifies the information criterion used to compute info values, default is 'aic'.
    iota : float, optional
        Small constant to ensure numerical stability in log calculations.
    eps : float, optional
        Small constant to avoid division by zero in calculations.
    proj : nn.Module, optional
        Optional final projection of aggregated vertex features and variances.

    Returns
    -------
    TokenizerFeatures
        The result of the tokenization.
    '''    
    # Two connected component calls to do transitive closure
    cc, nV = connected_components(state.E[0], state.E[1], state.nV)
    E = update_edges(cc, state.E, nV)
    ccn, nV = connected_components(E[0], E[1], nV)
    cc = ccn[cc]
    curseg = cc[state.curseg]
    fV, s2, mV = update_params(cc, state.fV, state.s2, state.mV, nV)
    info = infocrit(s2, mV, state.grid.H, state.grid.W, criterion, iota, eps)
    optinfo, optseg, optfV, opts2, optnV, _ = update_info(
        curseg, info, state.optseg, state.optinfo, 
        fV, state.optfV, s2, state.opts2, 
        state.optnV, 1,
    )

    # Finalize optimal segmentation based on infocrit
    u,v = origE[:,torch.eq(*optseg[origE])]
    seg, nV = connected_components(u, v, state.grid.B * state.grid.H * state.grid.W, maxit=100)
    uidx = optseg[fast_uidx_1d(seg * nV + optseg)]
    fV = optfV[uidx]
    s2 = opts2[uidx]
    grid = state.grid
    bb = torch.stack([grid.byx[1],grid.byx[2],grid.byx[1],grid.byx[2]], 0)
    assert bb is not None
    bb = update_bbox(seg, bb, nV)
    assert bb is not None
    return TokenizerFeatures(proj(fV), seg, nV, bb, grid)


def token_drop_merge(
    res:TokenizerFeatures, similarity:str='gaussian', 
    target:Optional[int]=None, boost:float=1.5,
    stochastic:bool=True
) -> TokenizerFeatures:
    '''Drops tokens with merging.

    Drops features with probability inversely proportional to size. 
    Used to limit number of tokens for training / for more consistent results.

    Parameters
    ----------
    res : TokenizerFeatures
        The final pruned tokenizer features.
    similarity : str, optional
        Similarity kernel.
    target : int, optional
       Target no. tokens for dropout. 
    boost : float, optional.
        Scalar for boosting xor drop values. 
        NOTE: Encourages merges btw. dropped and non-dropped tokens.

    Returns
    -------
    TokenizerFeatures
        The result of the merged tokenization.
    '''
    if target is None:
        target = int(round((res.grid.H * res.grid.W)**.5))
    
    mV = res.seg.bincount()
    E = get_seg_edges(res.seg, res.grid.B, res.grid.H, res.grid.W, True)
    mask, b_idx, bc, st = _get_token_mask(res.seg, res.grid.byx[0], res.grid.B, 1 / mV.sqrt())
    
    # Check for valid merge target
    if mask.shape[-1] <= target:
        return res
    
    if stochastic:
        samples = mask.multinomial(target)
    else:
        samples = torch.sort(mask, -1, descending=True)[1][:,:target]

    # Get masks for drops
    select = (samples + st.view(-1,1))[samples < bc.view(-1,1)]
    nkeepmask = torch.ones_like(b_idx, dtype=torch.bool)
    nkeepmask[select] = False
    loop = torch.eq(*E)
    drop_mask = torch.logical_or(*(nkeepmask)[E])
    xor_mask = torch.logical_xor(*(nkeepmask)[E])

    # Compute edge potentials and prune
    fE = res.fV.new_zeros(E.shape[1])
    fE[drop_mask] = get_similarity(similarity)(res.fV, E[:,drop_mask])
    fE[xor_mask].mul_(boost).clip(0,1)
    fE[loop & ~drop_mask] = 1
    fE[loop & drop_mask] = 0

    # Argmax and connected components
    amx = scatter_bitpack_argmax_edges(fE, E[0], E[1], res.nV)
    cc, nV = connected_components(torch.arange(res.nV, device=amx.device), amx, res.nV)

    # Update all parameters
    seg = cc[res.seg]
    bb = update_bbox(cc, res.bb, nV)
    num_w = mV.view(-1,1)
    den_w = scatter_sum_2d(mV, cc, nV).view(-1,1)
    fV = scatter_sum_2d(num_w * res.fV, cc, nV) / den_w
    assert bb is not None
    return TokenizerFeatures(fV, seg, nV, bb, res.grid)

