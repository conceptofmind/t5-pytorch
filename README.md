## T5 - PyTorch (WIP)
A PyTorch implementation of [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683). You can find the official T5x repository by Google [here](https://github.com/google-research/t5x).

There is a small bug with dimensionality which needs to be resolved.

## Acknowledgement

Phil Wang (lucidrains) advised and provided review for this implementation. [Please be sure to follow and support his work](https://github.com/lucidrains?tab=repositories).

## Usage

```python
import torch
from t5_pytorch import T5

model = T5(
    dim = 768,
    enc_num_tokens = 512,
    enc_depth = 6,
    enc_heads = 12,
    enc_dim_head = 64,
    enc_mlp_mult = 4,
    dec_num_tokens = 512,
    dec_depth = 6,
    dec_heads = 12,
    dec_dim_head = 64,
    dec_mlp_mult = 4,
    dropout = 0.,
    tie_token_emb = True
)

src = torch.randint(0, 512, (1, 1024))
src_mask = torch.ones_like(src).bool()
tgt = torch.randint(0, 512, (1, 1024))

output = model(src, tgt, mask = src_mask)

print(output.shape) #torch.Size([1, 1024, 512])
```

## Abstract

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new ``Colossal Clean Crawled Corpus'', we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code.


## Citations

```bibtex
@misc{https://doi.org/10.48550/arxiv.1910.10683,
  doi = {10.48550/ARXIV.1910.10683},
  
  url = {https://arxiv.org/abs/1910.10683},
  
  author = {Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J.},
  
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer},
  
  publisher = {arXiv},
  
  year = {2019},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
