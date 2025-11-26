import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Literal

from .mlp import MLP
from .att import MSA
from ..tok.tokenizer import dHTTokenizer
from ..tok.extractor import dHTExtractor
from ..tok.embedder import dHTEmbedder, dHTResult
from ..utils.denseloss import MaskedTargets

from ..utils.objtools import update_signature_kwargs


class TransformerBlock(nn.Module):

    kwarg_list = [
        'mlp_ratio', 'dop_path', 
    ]
    
    def __init__(
        self, embed_dim, heads, mlp_ratio=4.0, dop_path:float=0.0, 
        init_scale=1e-1, scale_by_keep=True, actfn='gelu', 
        qkv_bias:bool=True, qk_norm:bool=False, num_cls_tokens:int=1,
        **kwargs
    ):
        super().__init__()
        self.dop_path = dop_path
        self.scale_by_keep = scale_by_keep

        self.embed_dim = embed_dim
        self.heads = heads

        self.ls1 = nn.ParameterDict({
            'lambda_': nn.Parameter(torch.full((embed_dim,), init_scale))
        })
        self.ls2 = nn.ParameterDict({
            'lambda_': nn.Parameter(torch.full((embed_dim,), init_scale))
        })
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.attn = MSA(embed_dim, heads, qkv_bias, qk_norm)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), actfn=actfn)
        
    def sample_dop_masks(self, amask) -> Tensor:
        if amask.ndim != 2:
            raise NotImplementedError('Stoch.depth only for amask.ndim == 2!')
        B, M = amask.shape
        kwargs = {'dtype':bool, 'device':amask.device}
        masks = torch.empty(2, B, **kwargs).bernoulli_(1-self.dop_path)
        return masks.view(2,B,1,1) * amask.view(1,B,M,1)
        
    def forward(self, x, seg, amask, *args, **kwargs):
        if self.dop_path > 0 and self.training:
            m1, m2 = self.sample_dop_masks(amask)
        else:
            m1 = m2 = 1

        sc1, sc2 = self.ls1.lambda_, self.ls2.lambda_
        if self.training and self.scale_by_keep:
            sc1, sc2 = sc1 / (1-self.dop_path), sc2 / (1-self.dop_path)
        
        x = x + sc1 * (m1 * self.attn(self.norm1(x), amask))
        x = x + sc2 * (m2 * self.mlp(self.norm2(x), amask=amask, **kwargs))

        return x, seg, amask
    

class dHTEncoder(nn.Module):

    _capacities = {
        'S': {'depth':12, 'embed_dim': 384, 'heads': 6, 'dop_path':0.0},
        'M': {'depth':12, 'embed_dim': 512, 'heads': 8, 'dop_path':0.1},
        'B': {'depth':12, 'embed_dim': 768, 'heads':12, 'dop_path':0.2},
        'L': {'depth':24, 'embed_dim':1024, 'heads':16, 'dop_path':0.2},
        'H': {'depth':32, 'embed_dim':1280, 'heads':16, 'dop_path':0.2},
    }

    def __init__(
        self, embed_dim, patch_size, heads, depth, 
        channels=3, num_cls_tokens=1, tokenizer_hidden=3, 
        compute_grad=False,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.heads = heads
        self.depth = depth
        self.channels = channels
        self.num_cls_tokens = num_cls_tokens
        self.tokenizer_hidden = tokenizer_hidden
        self.compute_grad = compute_grad
 
        self.tokenizer = dHTTokenizer(
            channels, tokenizer_hidden, 
            **update_signature_kwargs(
                dHTTokenizer, compute_grad=compute_grad, **kwargs
            )
        )

        self.extractor = dHTExtractor(
            patch_size,
            **update_signature_kwargs(
                dHTExtractor, channels=channels, **kwargs
            )
        )

        self.embedder = dHTEmbedder(
            embed_dim, patch_size,
            **update_signature_kwargs(
                dHTEmbedder, channels=channels, compute_grad=compute_grad, 
                num_cls_tokens=num_cls_tokens, **kwargs
            )
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, heads, 
                **update_signature_kwargs(
                    TransformerBlock, 
                    num_cls_tokens=num_cls_tokens, 
                    **kwargs
                ),
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)


    def tokenizer_pipeline(self, img, final_merging, target) -> dHTResult:
        return self.embedder(
            self.extractor(
                self.tokenizer(
                    img, final_merging=final_merging, target=target
                )
            )
        )

    def forward(self, x, final_merging=True, target=None):
        x, seg, amask = self.tokenizer_pipeline(x, final_merging, target)
        
        for i, block in enumerate(self.blocks):
            x, seg, amask = block(x, seg, amask)

        return dHTResult(
            self.norm(x), seg, amask
        )
    
    @classmethod
    def build(cls, capacity:str, patch_size:int, **kwargs):
        construct_kwargs = cls._capacities.get(capacity.upper())
        valid = ', '.join(list(cls._capacities.keys()))
        if construct_kwargs is None:
            raise ValueError(f'Invalid capacity: {capacity.upper()}.\nValid capacities: {valid}')
        construct_kwargs['patch_size'] = patch_size
        construct_kwargs.update(kwargs)
        return cls(**construct_kwargs)
    
    def freeze_blocks(self):
        for p in self.blocks.parameters():
            p.requires_grad_(False)

    def unfreeze_blocks(self):
        for p in self.blocks.parameters():
            p.requires_grad_(True)

    def expand_with_grad(self, eps=1e-5):
        self.embedder.expand_with_grad(eps=eps)
        self.tokenizer.expand_with_grad()

    def interpolate_pos_embed(self, new_pos_patch_size, pos_patch_scale=None):
        self.extractor.interpolate_pos_embed(new_pos_patch_size, pos_patch_scale=pos_patch_scale)
        self.embedder.interpolate_pos_embed(new_pos_patch_size)


class dHTClassifier(dHTEncoder):

    def __init__(self, embed_dim, patch_size, heads, depth, n_classes=1000, **kwargs):
        super().__init__(embed_dim, patch_size, heads, depth, **kwargs)
        self.n_classes = n_classes
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, final_merging=True, target=None, encode_only=False):
        x, seg, amask = super().forward(x, final_merging, target)
        if encode_only:
            return x, seg, amask
        return self.head(x[:,0])

    

class dHTDensePretrainer(dHTEncoder):

    def __init__(self, embed_dim, patch_size, heads, depth, n_classes=201, mlp_head=True, **kwargs):
        super().__init__(embed_dim, patch_size, heads, depth, **kwargs)
        self.n_classes = n_classes
        mlp_ratio = kwargs.get('mlp_ratio', 4.0)
        actfn = kwargs.get('actfn', 'gelu')
        if mlp_head:
            self.head = nn.Sequential(
                MLP(self.embed_dim, int(self.embed_dim * mlp_ratio), actfn=actfn),
                nn.Linear(self.embed_dim, self.n_classes)
            )
        else:
            self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, final_merging=True, target=None, encode_only=False):
        x, seg, amask = super().forward(x, final_merging, target)
        if encode_only:
            return x, seg, amask
        return self.head(x), seg, amask
    
    def replace_head_with_mlp(self, mlp_ratio=4.0, actfn='gelu'):
        self.head = nn.Sequential(
            MLP(self.embed_dim, int(self.embed_dim * mlp_ratio), actfn=actfn),
            nn.Linear(self.embed_dim, self.n_classes)
        )
    
    def replace_head_with_linear(self):
        self.head = nn.Linear(self.embed_dim, self.n_classes)


class dHTDenseModel(dHTDensePretrainer):

    def __init__(self, embed_dim, patch_size, heads, depth, n_classes=201, concat_last=1, concat_mask=None, mlp_head=True, **kwargs):
        super().__init__(embed_dim, patch_size, heads, depth, **kwargs)
        self.n_classes = n_classes
        if concat_mask is None:
            concat_last = max(1,concat_last)
            self.concat = [concat_last - depth + i >= 0 for i in range(depth)]
        else:
            assert len(concat_mask) == depth
            assert all([isinstance(c, bool) for c in concat_mask])
            self.concat = concat_mask
        mlp_ratio = kwargs.get('mlp_ratio', 4.0)
        actfn = kwargs.get('actfn', 'gelu')
        hidden = int(concat_last * self.embed_dim)
        if hidden > self.embed_dim:
            self.norm = nn.LayerNorm(hidden)
        if mlp_head:
            self.head = nn.Sequential(
                MLP(hidden, int(hidden * mlp_ratio), actfn=actfn),
                nn.Linear(hidden, self.n_classes)
            )
        else:
            self.head = nn.Linear(hidden, n_classes)

        self.masked_targets = MaskedTargets(n_classes)

    def forward(self, x, final_merging=True, target=None, labels=None):
        x, seg, amask = self.tokenizer_pipeline(x, final_merging, target)
        
        concat:Optional[list[Tensor]] = []
        if sum(self.concat) <= 1:
            concat = None 

        for i, block in enumerate(self.blocks):
            x, seg, amask = block(x, seg, amask)
            if self.concat[i] and concat is not None:
                concat.append(x)
        if concat is not None:
            x = torch.cat(concat, -1)

        x = self.norm(x)

        if labels is not None:
            assert amask is not None
            labels = self.masked_targets(seg, labels)
            nc = self.num_cls_tokens
            x = x[:,nc:][amask[:,nc:]]
            return x, labels
        
        return self.head(x), seg, amask
