### About
This is a reference implementation of an Mixture-of-Experts Transformer (MoE) model in Jax and Haiku. The original motivation was to personally learn Jax and to understand the details within MoE. I've documented the implementation so that it could be helpful for others who are going through a similar process.

### Key Features
The techniques used are fairly standard among state-of-the-art models. So far it supports:
- Parallel implementation of the Byte Pair Encoding (BPE) algorithm. 
- Grouped Query Attention (GQA)
  - [paper](https://arxiv.org/pdf/2305.13245)
  - A generalization of Multi-Query Attention, where multiple query heads share a single key and value head. In GQA there are multiple key and value heads, each shared by a _group_ of query heads. Attention is computed between each key/value head and the query heads within the group. The results for all query heads across all groups are then concatenated and projected down to a lower dimension by the projection matrix (the same way as in multi-head attention.)
- Rotary Embedding
  - [paper](https://arxiv.org/pdf/2104.09864)
- Use KV cache to keep the computed key and value heads for previous tokens.
- Mixture-of-Experts (MoE) block with capacity
  - We use a token-choice model where each token chooses top K experts computed by the router.
  - Tokens that fall outside the capacity of an expert will be passed through as a residual connection.
  - Computatin for each expert is parallelized, potentially on separate devices.

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
- [minBPE](https://github.com/rsennrich/minbpe)