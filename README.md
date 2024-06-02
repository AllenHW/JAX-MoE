### About
This is a reference implementation of an Mixture-of-Experts Transformer (MoE) model in Jax and Haiku. The original motivation was for myself to learn Jax and to understand the details within MoE. I've made efforts to document the implementaton so that it could be helpful for others who are going through a similar process. 

### Key Features
The techniques used are fairly standard among state of the art models. So far it supports:
- Grouped Query Attention (GQA)
  - [paper](https://arxiv.org/pdf/2305.13245)
  - A generalization of Multi-Query Attention, which supports having mutiple query heads look up against a single key head and bring in embeddings from a single value head.
- Rotary Embedding
  - [paper](https://arxiv.org/pdf/2104.09864)
- Use KV cache to keep the computed key and value heads for previous tokens.
- Mixture-of-Experts (MoE) block with capacity
  - The current implementation uses a token choice model where each token are chhooses top K experts to which it was assigned by the router.
  - Tokens that fall outside the capacity of an expert will be passed through as residual connection (i.e. effectively handled by a "no-op" expert)
  - Computatin for each expert can be done on separate devices.
- High specificity of hyperparameters. Most implementations I've found generally assumed certain parameters to be coupled - for example, the key dimenion to be the model dimension divided by the number of heads. While such choice is proabably reasonable, for the sake of completeness I've made minmal assumptions on the relationships between these parameters so that the full space of hyperparameters can be explored.
- Optimized for JIT compilation.
  - More to explore here but it's matching the official implementations in Haiku.
- Parallel Byte Pair Encoding (BPE) algorithm

### TODO
- [ ] Implement training loop
- [ ] Find a dataset to use for training
- [ ] Generation code
- [ ] Large scale training, potentially on multi GPUs
- [ ] Implement Flash Attention??
- [ ] Attempt some intrepretability techniques

### References
- [Illusrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Switch Transformer](https://arxiv.org/pdf/2101.03961)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Grok-1](https://github.com/xai-org/grok-1)
- [Jax Tutorial](https://jax.readthedocs.io/en/latest/tutorials.html)
- [Mistral](https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/model.py)
- [Multi-Query Attention](https://arxiv.org/pdf/1911.02150)
- [Rotary Embedding](https://arxiv.org/pdf/2104.09864)