import jax.numpy as jnp
from typing import Dict, List
from jax import grad, jit, vmap, random
import jax 
import haiku as hk

"""
TODO:
- make things jittable and graddable

- tokenizer 

- training
- multi GPU
"""

w_init = hk.initializers.TruncatedNormal(stddev=1)


class Dropout(hk.Module):
  def __init__(self, rate: float = 0.1, name: str=None) -> None:
    super().__init__(name)
    self.rate = rate

  def __call__(self, x: jnp.array, is_training=True) -> jnp.array:
    if is_training:
      key = random.split(random.key(1), 1)
      # apply the scaling during training so inference is faster
      dropout_mask = random.bernoulli(key, 1 - self.rate, x.shape) / (1 - self.rate)
      return x * dropout_mask
    else:
      return x


class Linear(hk.Module):
  def __init__(self, 
              in_dim: int, 
              out_dim: int, 
              bias: bool = True,
              name: str = None) -> None:
    super().__init__(name)
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.bias = bias

  def __call__(self, x: jnp.array) -> jnp.array:
      w = hk.get_parameter('w', (self.in_dim, self.out_dim), init=w_init)
      b =  hk.get_parameter('b', (self.out_dim,), init=w_init)
      ret = jnp.einsum('io, ...i -> ...o',  w, x)
      if self.bias:
        ret += b
      return ret


class RMSNorm(hk.Module):
  def __init__(self, 
              eps: float = 1e-5, 
              name: str = None) -> None:
    super().__init__(name)
    self.eps = eps

  def __call__(self, x: jnp.array) -> jnp.array:
    input_dtype = x.dtype
    x = x.astype(jnp.float32)

    mean_squared = jnp.mean(jnp.square(x), axis=[-1], keepdims=True)
    x = x / jnp.sqrt(mean_squared + self.eps)
    scale =  hk.get_parameter('scale', (1,), init=w_init)
    x = scale * x
  
    return x.astype(input_dtype)



class DenseFF(hk.Module):
  def __init__(self,
              emd_dim: int, 
              hidden_dim: int, 
              activation: str = 'gelu', 
              bias: bool = True,
              name: str = None) -> None:
    super().__init__(name)
    self.w1 = Linear(emd_dim, hidden_dim, bias=bias)
    self.w2 = Linear(emd_dim, hidden_dim, bias=bias)
    self.w3 = Linear(hidden_dim, emd_dim, bias=bias)

    if activation == 'gelu':
      self.activation = jax.nn.gelu
    elif activation == 'silu':
      self.activation = jax.nn.silu
    else:
      raise ValueError(f'Unknown activation function: {activation}')

  def __call__(self, x: jnp.array) -> jnp.array:
    return self.w3(self.activation(self.w2(x)) * self.w1(x))



class RotaryEmbedding(hk.Module):
  def __init__(self, 
              dim, 
              base: int = 10000,
              name: str = None) -> None:
    super().__init__(name)
    assert dim % 2 == 0

    exps = -jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    self.thetas = base ** exps
    self.cached_seq_len = 0
    self.cos_cache = jnp.zeros((0,0,0,0))
    self.sin_cache = jnp.zeros((0,0,0,0))

  def _neg_half(self, x: jnp.array) -> jnp.array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.stack([-x2, x1], axis=-1)

  def __call__(self, x: jnp.array, offset: int = 0) -> jnp.array:
    """
      x: B x t x H x D
    """
    _, seq_len, _, _ = x.shapes 
    if self.cached_seq_len >= offset + seq_len:
      # cos, sin: 1 x t x 1 x D
      cos = self.cos_cache[offset:offset + seq_len]
      sin = self.sin_cache[offset:offset + seq_len]
      rote = x * cos + self._neg_half(x) *sin
    if self.cached_seq_pos < offset + seq_len:
      t = jnp.arange(self.cached_seq_pos, offset + seq_len, dtype=jnp.float32)
      # ticks: t x D/2
      ticks = jnp.outer(t, self.thetas)
      # ticks: t x D/2 -> 1 x t x 1 x D 
      ticks = jnp.tile(ticks, reps=(1,2))[None, :, None :] 

      # cos_cache, sin_cache: 1 x t' x 1 x D
      self.cos_cache = jnp.stack(self.cos_cache, jnp.cos(ticks), axis=1)
      self.sin_cache = jnp.stack(self.sin_cache, jnp.sin(ticks), axis=1)
      self.cached_seq_len = offset + seq_len

    cos = self.cos_cache[offset:offset + seq_len]
    sin = self.sin_cache[offset:offset + seq_len]
    rote = x * cos + self._neg_half(x) *sin
    return rote


class MultiHeadSelfAttention(hk.Module):
    def __init__(self, 
        num_q_heads, 
        num_kv_heads,
        emd_dim: int,
        v_dim: int,
        k_dim: int,
        bias: bool = False,
        att_dropout: float = 0.1,
        resid_dropout: float = 0.0,
        name: str = None,
    ) -> None:
        super().__init__(name)
        assert num_q_heads % num_kv_heads == 0
        assert att_dropout >= 0 and resid_dropout >=0

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.q_group_size = num_q_heads // num_kv_heads
        self.emd_dim = emd_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.bias = bias

        self.wq = Linear(self.emd_dim, self.k_dim * self.num_q_heads, bias=self.bias)
        self.wk = Linear(self.emd_dim, self.k_dim * self.num_kv_heads, bias=self.bias)
        self.wv = Linear(self.emd_dim, self.v_dim * self.num_kv_heads, bias=self.bias)
        self.wo = Linear(self.q_group_size * self.num_kv_heads * self.v_dim, self.emd_dim, 
          bias=self.bias
        )

        self.attn_dropout = Dropout(rate=att_dropout)
        self.resid_dropout = Dropout(rate=resid_dropout)
        self.rote = RotaryEmbedding(self.emd_dim)

    def __call__(self, x: jnp.array, kv_cache: jnp.array = None, is_training=True) -> jnp.array:
      """
        B: batch size
        t: length of the input sequence
        T: length of the attended sequence
        D: embedding dimension
        K=Q: key/query dimension
        V: value dimension
        h: number of query heads
        H: number of key/value heads
        G: number of query groups
      """

      # x: B x t x D
      # kv_cache: B x T x H x (K+V)
      # q_heads: B x t x h x Q=K
      # k_heads: B x T x H x K
      # v_heads: B x T x H x V
      q_heads, k_heads, v_heads = self._attention_heads(x, kv_cache)
      # attn_output: B x t x D
      attn_output = self._grouped_attention(q_heads, k_heads, v_heads, is_training)

      new_kv_cache = jnp.concatenate([k_heads, v_heads], axis=-1)
      
      return attn_output, new_kv_cache


    def _grouped_attention(self, q_heads: jnp.array, k_heads: jnp.array, v_heads: jnp.array, is_training) -> jnp.array:
      b_size, q_seq_len, _, _,  = q_heads.shape
      _, k_seq_len, _, _ = k_heads.shape
      
      # grouped_q_heads: B x t x h x K -> B x t x G x H x K
      grouped_q_heads = q_heads.reshape((b_size, q_seq_len, self.q_group_size, self.num_kv_heads, self.k_dim))

      # Dot product of query and key heads. G = query group size. Aggregation over K = key dimension. 
      attn_scores = jnp.einsum('BtGHK, BTHK-> BGHtT', grouped_q_heads, k_heads).astype(jnp.float32) 
      # Mask out future tokens. Assume that query tokens are suffix of the key tokens. Take ending rows of the LT matrix.
      attn_mask = jnp.tril(jnp.ones((k_seq_len, k_seq_len)))[-q_seq_len:, :]
      attn_scores = jnp.where(attn_mask, attn_scores, -jnp.inf)
      # Softmax, followed by dropout
      attn_scores = jax.nn.softmax(attn_scores / jnp.sqrt(self.k_dim), dim=-1).astype(v_heads.dtype)
      attn_scores = self.attn_dropout(attn_scores, is_training)
      # Apply attention scores on the value heads. Aggregation over T = length of the attended sequence
      attn_output = jnp.einsum("BGHtT, BTHV -> BtGHV", attn_scores, v_heads)

      # Reshape attn_output: B x t x G x H x V ->  B x t x (G * H * V)
      attn_output = attn_output.reshape(b_size, q_seq_len, self.q_group_size * self.num_kv_heads * self.v_dim)
      # Apply linear map on the attn_output: B x t x (G * H * V) -> B x t x D, followed by dropout
      attn_output = self.wo(attn_output)
      attn_output = self.resid_dropout(attn_output, is_training)

      return attn_output

    def _attention_heads(self, x: jnp.array, kv_cache: jnp.array = None) -> List[jnp.array]:
      """
        x: B x t x D
        kv_cache: B x T x H x (K+V)
      """
      b_size, seq_len, _ = x.shape
      offset = 0 if kv_cache is None else kv_cache.shape[1]

      # q_heads: B x t x h x Q=K
      q_heads = self.wq(x) \
                .reshape(b_size, seq_len, self.num_q_heads, self.k_dim)
      q_heads = self.rote(q_heads, offset) 
      # k_heads: B x T x H x K
      k_heads = self.wk(x) \
                .reshape(b_size, seq_len, self.num_kv_heads, self.k_dim)
      k_heads = self.rote(k_heads, offset)
      # v_heads: B x T x H x V
      v_heads = self.wv(x) \
                .reshape(b_size, seq_len, self.num_kv_heads, self.v_dim)
      
      if kv_cache:
        b_size_cached, seq_len_cached, num_kv_heads_cached, kv_dim_cached = kv_cache.shape
        assert (b_size_cached, num_kv_heads_cached, kv_dim_cached) == (b_size, self.num_kv_heads, self.k_dim + self.v_dim)
        
        k_heads_cached = kv_cache[:, :, :, :self.k_dim]
        v_heads_cached = kv_cache[:, :, :, self.k_dim:]

        k_heads = jnp.concatenate([k_heads_cached, k_heads], axis=1)
        v_heads = jnp.concatenate([v_heads_cached, v_heads], axis=1)

      return q_heads, k_heads, v_heads
    


class MoEBlock(hk.Module):
  def __init__(self,
      emd_dim: int,
      experts: List[DenseFF],
      active_experts: int = 2,
      name: str = None
  ):
    super().__init__(name)
    self.top_k = active_experts
    self.experts = experts
    self.router = Linear(emd_dim, len(experts))

  def _compute_expert_scores(self, x: jnp.array) -> jnp.array:
    """
      x: B x t x D
      expert_scores: B x t x num_experts
      selected_experts: B x t x top_k
      
      B: batch size
      t: length of the input sequence
      D: embedding dimension
    """
    # expert_scores: B x t x num_experts
    expert_scores = self.router(x.astype(jnp.float32))
    expert_scores, selected_experts = jax.lax.top_k(expert_scores, k=self.top_k)
    expert_scores = jax.nn.softmax(expert_scores, axis=-1).astype(x.dtype)

    return expert_scores, selected_experts

  def __call__(self, x: jnp.array) -> jnp.array:
    """
      x: B x t x D
      
      B: batch size
      t: length of the input sequence
      D: embedding dimension
    """
    # expoert_scores: B x t x top_k
    expert_scores, selected_experts = self._compute_expert_scores(x)

    r = jnp.zeros_like(x)
    for i, expert in enumerate(self.experts):
      # select batch and token indices routed to this expert
      b, t, e = jnp.where(selected_experts == i)
      # apply expert to the selected tokens and update batch and token indices
      r = r.at[b, t].add(expert(x[b, t])) * expert_scores[b, t, e]

    return r


class TransformerBlock(hk.Module):
  def __init__(self, 
    emd_dim: int,
    hidden_dim: int,
    num_experts: int,
    active_experts: int,
    num_q_heads, 
    num_kv_heads,
    v_dim: int,
    k_dim: int,
    ff_bias: bool = False,
    attn_bias: bool = False,
    att_dropout: float = 0.1,
    attn_resid_dropout: float = 0.0,
    name: str = None,
  ) -> None:
    super().__init__(name)
    self.pre_layer_norm = RMSNorm()
    self.post_attn_norm = RMSNorm()
    self.attn = MultiHeadSelfAttention(
      num_q_heads, 
      num_kv_heads,
      emd_dim,
      v_dim,
      k_dim,
      bias=attn_bias,
      att_dropout=att_dropout,
      resid_dropout=attn_resid_dropout
    )
    self.num_experts = num_experts
    if num_experts > 1:
      experts = [DenseFF(emd_dim, hidden_dim, bias=ff_bias) for _ in range(num_experts)]
      self.moe_block = MoEBlock(emd_dim, experts, active_experts)
    else:
      self.ff = DenseFF(emd_dim, hidden_dim, bias=ff_bias)
  def __call__(self, x: jnp.array, kv_cache: jnp.array = None, is_training=True) -> jnp.array:
    """
      x: B x t x D
      kv_cache: B x T x H x (K+V)
      
      B: batch size
      t: length of the input sequence
      D: embedding dimension
      K=Q: key/query dimension
      V: value dimension
      H: number of key/value heads
    """

    h = self.pre_layer_norm(x)
    h, new_kv_cache = self.attn(h, kv_cache, is_training)
    h = h + x 
    r = self.post_attn_norm(h)

    if self.num_experts > 1:
      r = self.moe_block(r)
    else:
      r = self.ff(r)
    r = r + h

    return r, new_kv_cache
  

class Embedding(hk.Module):
  def __init__(self, 
              emd_dim: int, 
              vocab_size: int,
              name: str = None) -> None:
    super().__init__(name)
    self.emd_dim = emd_dim
    self.vocab_size = vocab_size

  def encode(self, x: jnp.array) -> jnp.array:
    embedding = hk.get_parameter('embedding', (self.vocab_size, self.emd_dim), init=w_init)
    return jnp.einsum('ve, ...v -> ...e', embedding, x)

  def decode(self, x: jnp.array) -> jnp.array:
    embedding = hk.get_parameter('embedding', (self.vocab_size, self.emd_dim), init=w_init)
    return jnp.einsum('ve, ...e -> ...v', embedding, x)


class MoETransformer(hk.Module):
    def __init__(self,
      depth: int, 
      vocab_size: int,
      emd_dim: int,
      hidden_dim: int,
      num_experts: int,
      num_q_heads, 
      num_kv_heads,
      v_dim: int,
      k_dim: int,
      ff_bias: bool = False,
      attn_bias: bool = False,
      att_dropout: float = 0.1,
      attn_resid_dropout: float = 0.0,
      name: str = None,
    ): 
      super().__init__(name)

      self.blocks = [
        TransformerBlock(
          emd_dim,
          hidden_dim,
          num_experts,
          num_q_heads, 
          num_kv_heads,
          v_dim,
          k_dim,
          ff_bias,
          attn_bias,
          att_dropout,
          attn_resid_dropout
        )
        for _ in range(depth)
      ]
      self.embedding = Embedding(emd_dim, vocab_size)
      self.normalize = RMSNorm()
      

    def __call__(self, x: jnp.array, kv_caches: Dict[int, jnp.array] = {}, is_training=True) -> jnp.array:
      """
        x: B x t x D
        kv_caches[i]: B x T x H x (K+V)
        
        B: batch size
        t: length of the input sequence
        T: length of the attended sequence
        D: embedding dimension
        K=Q: key/query dimension
        V: value dimension
        H: number of key/value heads
      """
      new_kv_caches = {}

      h = self.embedding.encode(x)
      
      for i, block in enumerate(self.blocks):
        h, new_kv_cache = block(h, kv_caches[i], is_training)
        new_kv_caches[i] = new_kv_cache

      h = self.normalize(h)
      r = self.embedding.decode(h)

      return r, new_kv_caches