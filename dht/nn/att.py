import torch
import torch.nn as nn
import torch.nn.functional as F


class MSA(nn.Module):
    def __init__(
        self, embed_dim:int, heads:int, qkv_bias:bool=True, qk_norm:bool=False, **kwargs
    ):
        super().__init__()
        assert embed_dim % heads == 0, f'Invalid args: embed_dim % heads != 0.'
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim ** -.5
        self.qkv = nn.Linear(embed_dim, 3*embed_dim, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self._stored_k = None
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _getmsk(self, mask):
        if mask.ndim == 4:
            return mask
        if mask.ndim == 2:
            return mask[:,None,None]
        raise ValueError(f'Got mask with ndim == {mask.ndim}.')
        
            
    def forward(self, x, amask, *args, **kwargs): 
        B, M, D = x.shape
        H, K = self.heads, self.head_dim

        q, k, v = (
            self.qkv(x)
                .view(B, M, 3, H, K)
                .permute(2,0,3,1,4)
        )

        if kwargs.get('store_k', False):
            self._stored_k = k.permute(0,2,1,3).view(B,k.shape[2], -1)

        return self.proj(
            F.scaled_dot_product_attention(
                self.k_norm(q), self.q_norm(k), v, attn_mask=self._getmsk(amask)
            ).transpose(1,2).reshape(B,-1,D),
        )


class ManualMSA(MSA):


    def _getmsk(self, mask):
        mask = 1 - super()._getmsk(mask)
        return -mask * torch.inf

    def forward(self, x, amask, *args, **kwargs):
        B, M, D = x.shape
        H, D = self.heads, self.head_dim

        q, k, v = (
            self.qkv(x)
                .view(B, M, 3, H, D)
                .permute(2,0,3,1,4)
        )

        if kwargs.get('store_k', False):
            self._stored_k = k.permute(0,2,1,3).view(B,k.shape[2],-1)

        q = self.q_norm(q) * self.scale
        k = self.k_norm(k)
        attn = ((q @ k.mT) * self._getmsk(amask)).softmax(-1)
        if kwargs.get('store_attn', False):
            self._stored_attn = attn

        return self.proj(
            (attn @ v).transpose(1,2).reshape(B,-1,D)
        )
