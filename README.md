<div align="center">

# $\text{Differentiable Hierarchical Visual Tokenization}$

**[Marius Aasan](https://www.mn.uio.no/ifi/english/people/aca/mariuaas/)$^1$, [Martine Hjelkrem-Tan](https://www.mn.uio.no/ifi/english/people/aca/matan/)$^1$, [Nico Catalano](https://nicocatalano.github.io/)$^2$, [Chankyu Choi](https://en.uit.no/ansatte/person?p_document_id=617277)$^3$, [Adín Ramírez Rivera](https://www.mn.uio.no/ifi/english/people/aca/adinr/)$^1$** <br>


**${}^1\underset{\text{Department of Informatics}}{\text{University of Oslo}}$** $\hspace{1em}$ 
**${}^2\underset{\text{Artificial Intelligence and Robotics Lab}}{\text{Polytechnic University of Milan}}$** $\hspace{1em}$ 
**${}^3\underset{\text{Department of Physics and Technology}}{\text{UiT The Arctic University of Norway}}$** 
<br>

[![Website](https://img.shields.io/badge/Website-green)](https://dsb-ifi.github.io/dHT/)
[![PaperArxiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2507.01654)
[![PaperNeurIPS](https://img.shields.io/badge/Paper-NeurIPS_2025-blue)](https://eclr-workshop.github.io/)
[![SpotlightNeurIPS](https://img.shields.io/badge/Spotlight-NeurIPS_2025-cyan)](https://neurips.cc/virtual/2025/poster/115180)
[![NotebookExample](https://img.shields.io/badge/Notebook-Example-orange)](https://nbviewer.jupyter.org) <br>

![dHT Figure 1](/assets/dht_teaser_transparent.png#gh-light-mode-only "Examples of feature trajectoreis with SPoT-ON")
![dHT Figure 1](/assets/dht_teaser_transparent_dark.png#gh-dark-mode-only "Examples of feature trajectoreis with SPoT-ON")

## $\text{Abstract}$
</div>
<div style="font-family: serif;">
Vision Transformers rely on fixed patch tokens that ignore the spatial and semantic structure of images. In this work, we introduce an end-to-end differentiable tokenizer that adapts to image content with pixel-level granularity while remaining backward-compatible with existing architectures for retrofitting pretrained models. Our method uses hierarchical model selection with information criteria to provide competitive performance in both image-level classification and dense-prediction tasks, and even supports out-of-the-box raster-to-vector conversion.
</div>


## $\partial\text{HT}$: Differentiable Hierarchical Visual Tokenization

This repo contains code and weights for **Differentiable Hierarchical Visual Tokenization**, accepted for NeurIPS 2025.

For an introduction to our work, visit the [project webpage](https://dsb-ifi.github.io/dHT/). 

## Installation

The repo can currently be installed as a package via:

```bash
# HTTPS
pip install git+https://github.com/dsb-ifi/dHT.git

# SSH
pip install git+ssh://git@github.com/dsb-ifi/dHT.git
```

## Loading models

You can load the Superpixel Transformer models easily via `torch.hub`:

```python
model = torch.hub.load(
    'dsb-ifi/dht', 
    'dht_vit_base_16_in1k',
    pretrained=True,
    source='github',
)
```

This will load the model and downloaded the pretrained weights, stored in your local `torch.hub` directory. 

## More Examples

We provide a [Jupyter notebook](https://nbviewer.jupyter.org/) as a sandbox for loading, evaluating, and extracting token placements for the models. 

## Citation

If you find our work useful, please consider citing our paper.

```
@inproceedings{aasan2025dht,
  title={Differentiable Hierarchical Visual Tokenization},
  author={Aasan, Marius and Hjelkrem-Tan, Martine and Catalano, Nico and Choi, Changkyu and Ram\'irez Rivera, Ad\'in},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=y8VWYf5cVI}
}
```
