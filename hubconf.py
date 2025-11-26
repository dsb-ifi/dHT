'''
PyTorch Hub configuration for dHT models.
'''

dependencies = ['torch', 'torchvision', 'numpy', 'PIL']

from dht.models import load_dht_ras2vec as _load_dht_ras2vec

def dht_ras2vec(
    pretrained=True, 
    th0=0., 
    th1=0.1, 
    th2=0.1, 
    use_bg=True, 
    stroke=1.0, 
    patch_size=32,
    cmp=0.01, 
    iota=5.0,
    normalize=False,
    **kwargs
):
    '''Load dHT Raster-to-Vector (Ras2Vec) model.
    
    This model converts raster images to SVG vector representations using 
    differentiable hierarchical tokenization.
    
    Parameters
    ----------
    pretrained : bool, optional
        If True, load pretrained weights. Default: True
    th0 : float, optional
        Threshold for edge detection. Default: 0.0
    th1 : float, optional
        Threshold for token merging (low). Default: 0.1
    th2 : float, optional
        Threshold for token merging (high). Default: 0.1
    use_bg : bool, optional
        Include background in SVG output. Default: True
    stroke : float, optional
        Stroke width for SVG paths. Default: 1.0
    patch_size : int, optional
        Patch size for tokenization. Default: 32
    cmp : float, optional
        Compression parameter. Default: 0.01
    iota : float, optional
        Information criterion parameter. Default: 5.0
    normalize : bool, optional
        Apply ImageNet normalization. Default: False
        
    Returns
    -------
    dHTRas2Vec
        Loaded model ready for inference
        
    Examples
    --------
    >>> import torch
    >>> model = torch.hub.load('dsb-ifi/dHT', 'dht_ras2vec', pretrained=True, source='github')
    >>> # Load an image as a tensor (1, C, H, W) with ImageNet normalized values
    >>> img = ...  # Your image tensor
    >>> svg_string = model(img)
    '''
    return _load_dht_ras2vec(
        pretrained=pretrained,
        th0=th0,
        th1=th1,
        th2=th2,
        use_bg=use_bg,
        stroke=stroke,
        patch_size=patch_size,
        cmp=cmp,
        iota=iota,
        normalize=normalize
    )
