import os
import math
from functools import partial
DEVICE_COUNT = 8
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    DEVICE_COUNT
)

import jax.numpy as jnp  # noqa: E402
from typing import Dict, List # noqa: E402
from jax import grad, jit, vmap, random # noqa: E402
from jax.sharding import Mesh, PartitionSpec # noqa: E402
from jax.experimental import mesh_utils # noqa: E402
from jax.experimental.shard_map import shard_map # noqa: E402
import jax # noqa: E402
import haiku as hk # noqa: E402


"""
TODO:
- training loop
- generation
"""


w_init = hk.initializers.TruncatedNormal(stddev=1)

@partial(jit, static_argnames=('is_training', 'rate'))
def dropout(x: jax.Array, rate, rng, is_training=True):
  if is_training and rate > 0:
    dropout_mask = random.bernoulli(rng, 1 - rate, x.shape)
    return x * dropout_mask / (1 - rate)
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

  def __call__(self, x: jax.Array) -> jax.Array:
      w = hk.get_parameter('w', (self.in_dim, self.out_dim), init=w_init)
      ret = jnp.einsum('io, ...i -> ...o',  w, x)
      if self.bias:
        b = hk.get_parameter('b', (self.out_dim,), init=w_init)
        ret += b

      return ret

class RMSNorm(hk.Module):
  def __init__(self, 
              eps: float = 1e-5, 
              name: str = None) -> None:
    super().__init__(name)
    self.eps = eps

  def __call__(self, x: jax.Array) -> jax.Array:
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

    if activation not in ('gelu', 'silu', 'relu'):
      raise ValueError(f'Unknown activation function: {activation}')
      
    if activation == 'gelu':
      self.activation = jax.nn.gelu
    elif activation == 'silu':
      self.activation = jax.nn.silu
    elif activation == 'relu':
      self.activation = jax.nn.relu

  def __call__(self, x: jax.Array) -> jax.Array:
    h = self.w2(x) * self.w1(x)
    h = self.activation(h)
    return self.w3(h)


class RotaryEmbedding(hk.Module):
  def __init__(self, 
              dim,
              max_seq_len: int = 8000,
              base: int = 10000,
              name: str = None) -> None:
    super().__init__(name)
    assert dim % 2 == 0

    # thetas: D/2
    exps = -jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    thetas = base ** exps

    t = jnp.arange(0, max_seq_len, dtype=jnp.float32)
    # ticks: t x D/2
    ticks = jnp.outer(t, thetas)
    # ticks: t x D/2 -> 1 x t x 1 x D 
    ticks = jnp.tile(ticks, reps=(1,2))[None, :, None, :] 

    # cos, sin: 1 x t x 1 x D
    hk.set_state('cos', jnp.cos(ticks))
    hk.set_state('sin', jnp.sin(ticks))

  def _neg_half(self, x: jax.Array) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)

  def __call__(self, x: jax.Array, offset: int = 0) -> jax.Array:
    """
      x: B x t x H x D
    """
    _, seq_len, _, D = x.shape

    cos = jax.lax.dynamic_slice(hk.get_state('cos'), (0, offset, 0, 0), (1, seq_len, 1, D))
    sin = jax.lax.dynamic_slice(hk.get_state('sin'), (0, offset, 0, 0), (1, seq_len, 1, D))
    rote = x * cos + self._neg_half(x) * sin
    
    return rote


class MultiHeadAttention(hk.Module):
    def __init__(self, 
        num_q_heads, 
        num_kv_heads,
        emd_dim: int,
        v_dim: int,
        k_dim: int,
        bias: bool = False,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        name: str = None,
    ) -> None:
        super().__init__(name)
        assert num_q_heads % num_kv_heads == 0
        assert attn_dropout >= 0 and resid_dropout >=0

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

        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.rote = RotaryEmbedding(self.k_dim)

    # NOTE: Even for self attention, accepting three separate arguments is faster with jit
    def __call__(self, q: jax.Array, k: jax.Array, v: jax.Array, kv_cache: jax.Array = None, is_training=True) -> jax.Array:
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

        q, k, v: B x t x D
        kv_cache: B x T x H x (K+V)
      """

      # q_heads, k_heads: B x t x h x K
      # v_heads: B x T x H x V
      q_heads, k_heads, v_heads = self._attention_heads(q, k, v, kv_cache)
      attn_output = self._grouped_attention(q_heads, k_heads, v_heads, is_training) # B x t x D
      new_kv_cache = jnp.concatenate([k_heads, v_heads], axis=-1)
      
      return attn_output, new_kv_cache

    def _grouped_attention(self, q_heads: jax.Array, k_heads: jax.Array, v_heads: jax.Array, is_training) -> jax.Array:
      b_size, q_seq_len, _, _,  = q_heads.shape
      _, k_seq_len, _, _ = k_heads.shape

      grouped_q_heads = q_heads.reshape((b_size, q_seq_len, self.q_group_size, self.num_kv_heads, self.k_dim))

      attn_scores = jnp.einsum('BtGHK, BTHK-> BGHtT', grouped_q_heads, k_heads).astype(jnp.float32) 
      attn_mask = jnp.tril(jnp.ones((k_seq_len, k_seq_len)))[-q_seq_len:, :]
      attn_scores = jnp.where(attn_mask, attn_scores, -jnp.inf)

      attn_scores = jax.nn.softmax(attn_scores / jnp.sqrt(self.k_dim), axis=-1).astype(v_heads.dtype)
      attn_scores = dropout(attn_scores, self.attn_dropout, hk.next_rng_key(), is_training)

      attn_output = jnp.einsum("BGHtT, BTHV -> BtGHV", attn_scores, v_heads)
      attn_output = attn_output.reshape(b_size, q_seq_len, self.q_group_size * self.num_kv_heads * self.v_dim)

      attn_output = self.wo(attn_output)
      attn_output = dropout(attn_output, self.resid_dropout, hk.next_rng_key(), is_training)

      return attn_output

    def _attention_heads(self, q: jax.Array, k: jax.Array, v: jax.Array, kv_cache: jax.Array = None) -> List[jax.Array]:
      """
        q, k, v: B x t x D
        kv_cache: B x T x H x (K+V)
      """
      b_size, seq_len, _ = q.shape
      offset = 0 if kv_cache is None else kv_cache.shape[1]

      # q_heads: B x t x h x Q=K
      q_heads = self.wq(q).reshape(b_size, seq_len, self.num_q_heads, self.k_dim)
      q_heads = self.rote(q_heads, offset) 
      # k_heads: B x T x H x K
      k_heads = self.wk(k).reshape(b_size, seq_len, self.num_kv_heads, self.k_dim)
      k_heads = self.rote(k_heads, offset)
      # v_heads: B x T x H x V
      v_heads = self.wv(v).reshape(b_size, seq_len, self.num_kv_heads, self.v_dim)
      
      # stack newly computed heads with cached key and value heads to create the the full key and value heads
      if kv_cache is not None:
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
      hidden_dim: int,
      num_experts: int = 8,
      active_experts: int = 2,
      expert_capacity: float = 1.0,
      ff_bias: bool = False,
      multi_device=True,
      name: str = None
  ):
    super().__init__(name)
    if multi_device:
      assert num_experts <= DEVICE_COUNT or num_experts % DEVICE_COUNT == 0
    assert active_experts <= num_experts
    self.expert_capacity = expert_capacity
    self.top_k = active_experts

    
    self.emd_dim = emd_dim
    self.hidden_dim = hidden_dim
    self.ff_bias = ff_bias
    self.num_experts = num_experts
    self.router = Linear(emd_dim, num_experts)

    self.multi_device = multi_device
  
  def _compute_expert_scores(self, x_flat: jax.Array) -> jax.Array:
    expert_scores = self.router(x_flat.astype(jnp.float32)) # (B * t) x num_experts
    expert_scores, expert_assignment = jax.lax.top_k(expert_scores, k=self.top_k) # (B * t) x top_k
    expert_scores = jax.nn.softmax(expert_scores, axis=-1).astype(x_flat.dtype) # (B * t) x top_k

    return expert_assignment, expert_scores


  def _compute_token_expert_assignment(self, x_flat: jax.Array, expert_assignment: jax.Array) -> jax.Array:
    """
      B: batch size
      t: length of the input sequence
      D: embedding dimension

      x_flat: (B * t) x D
    """

    B_t, _ = x_flat.shape

    # expert capacity is the fraction of the total number of tokens that can be routed to an expert
    expert_capacity = math.floor(B_t * self.expert_capacity)
    # For each token, each choice, and  each expert, there is either an assignment of the token to the expert ornot
    expert_assignment_onehot = jax.nn.one_hot(expert_assignment, self.num_experts) # (B * t) x top_k x num_experts

    # Cumulative sum over the token indices, and then cumulaitive sum over the top_k dimension. This generates position of a token in an expert's capacity. The order of sum ensure that the first expert choice of a token is prioritized over the second expert choice of any other token, in the expert's position
    position_in_experts = jnp.cumsum(jnp.cumsum(expert_assignment_onehot, axis=0), axis=1).astype(jnp.int32) # (B * t) x top_k x num_experts
    
    # Using the assigned positions, remove any token that did not make it within an expert's capacity.
    expert_mask = expert_assignment_onehot * jnp.less(position_in_experts, expert_capacity) # (B * t) x top_k x num_experts
    # By summing over the experts dimenion, obtain as mask of which tokens are processed by some experts and which ones are orphants.
    token_assignment_mask = jnp.sum(expert_mask, axis=-1) # (B * t) x top_k


    # Apply onehot operate to the position array creates an occupancy mask for each token, choice index, expert, and the position in the expert. Multiply by the expert mask to remove tokens that were droppd
    expert_choices = jax.nn.one_hot(position_in_experts, expert_capacity) * expert_mask[..., None] # (B * t) x top_k x num_experts x expert_capacity
  
    # Sum over the top_k dimenion, because for each token, and, expert, the expert can only be chosen as one of the top_k choices - the resulting mask will be binary. It is an occupancy mask for each token, and for each capacity position, whether there's an allocation
    expert_choices = jnp.sum(expert_choices, axis=1) # (B * t) x num_experts x expert_capacity
    expert_choices = expert_choices.transpose(1, 2, 0).astype(jnp.int32) # num_experts x expert_capacity x (B * t)

    # For each token, and for each choice index, which expert it gets assigned to. Note that this map contains values of 0, which is ambiguous between assignment to expert 0 and the token being an orphant. But which tokens were dropped is store in token_assignment_mask
    expert_assignment = jnp.einsum('tke, e -> tk', expert_mask, jnp.arange(self.num_experts)) # (B * t) x top_k
    # Extract out the position in the selected expert
    position_in_selected_experts = jnp.einsum('tke, tke -> tk', expert_mask, position_in_experts) # (B * t) x top_k
    # Stack them to a single tensor
    expert_position_assignment = jnp.stack([expert_assignment, position_in_selected_experts], axis=-1).astype(jnp.int32) # (B * t) x top_k x 2


    return expert_choices, expert_position_assignment, token_assignment_mask

  def _compute_experts(self, grouped_x: jax.Array) -> jax.Array:
      def expert_fn(x):
        ff = DenseFF(self.emd_dim, self.hidden_dim, bias=self.ff_bias)
        return ff(x)
      
      # Initialize a batch of parameters
      expert_init, expert_apply = hk.transform(expert_fn)
      init_experts = hk.experimental.transparent_lift(
        vmap(expert_init, in_axes=0, out_axes=0),
        allow_reuse=True
      )
      expert_params = init_experts(
        hk.next_rng_keys(self.num_experts),
        jnp.zeros((self.num_experts, 1, self.emd_dim))
      )
      
      if self.multi_device:
        devices = mesh_utils.create_device_mesh(DEVICE_COUNT)[:self.num_experts]
        @partial(
          shard_map,
          mesh=Mesh(devices, axis_names=('e')),
          in_specs=(PartitionSpec('e',), PartitionSpec('e',)),
          out_specs=PartitionSpec('e',),
          check_rep=False
        )
        def parallel_expert_fn(params, grouped_x):
          return vmap(expert_apply, in_axes=(0, None, 0), out_axes=0)(params, None, grouped_x)
  
        expert_outputs = parallel_expert_fn(expert_params, grouped_x) # num_experts x expert_capacity x D
      else:
        expert_outputs = vmap(expert_apply, in_axes=(0, None, 0), out_axes=0)(expert_params, None, grouped_x) # num_experts x expert_capacity x D

      return expert_outputs

  def __call__(self, x: jax.Array) -> jax.Array:
    """      
      B: batch size
      t: length of the input sequence
      D: embedding dimension

      x: B x t x D
    """
    b, t, D = x.shape
    x_flat = x.reshape((b * t, D)) 

    # expert_assignment, expert_scores: (B * t) x top_k
    expert_assignment, expert_scores = self._compute_expert_scores(x_flat)

    # expert_choices: num_experts x expert_capacity x (B * t)
    # expert_position_assignment: (B * t) x top_k x 2
    # token_assignment_mask: (B * t) x top_k
    expert_choices, expert_position_assignment, token_assignment_mask = self._compute_token_expert_assignment(x_flat, expert_assignment)

    # Reshape to have a batch dimension
    expert_scores = jnp.reshape(expert_scores, ((b, t, self.top_k))) # B x t x top_k
    expert_position_assignment = jnp.reshape(expert_position_assignment, ((b, t, self.top_k, 2))) # B x t x top_k x 2
    token_assignment_mask = jnp.reshape(token_assignment_mask, ((b, t, self.top_k))) # B x t x top_k
    # Assign x into expert and capacity positions
    grouped_x = jnp.einsum('ect, tD -> ecD', expert_choices, x_flat) # num_experts x expert_capacity x D
    
    # Process x with each expert, potentially in parallel across the experts
    expert_outputs = self._compute_experts(grouped_x) # num_experts x expert_capacity x D
    expert_outputs = expert_outputs[expert_position_assignment[..., 0], expert_position_assignment[..., 1], :] # B x t x top_k x D
  
    # Dropped tokens will receive a residual connection 
    expert_outputs = jnp.where(token_assignment_mask[..., None], expert_outputs, x[..., None, :]) # B x t x top_k x D
    result = jnp.einsum('btkD, btk -> btD', expert_outputs, expert_scores) # B x t x D

    return result


class TransformerBlock(hk.Module):
  def __init__(self, 
    emd_dim: int,
    num_q_heads, 
    num_kv_heads,
    v_dim: int,
    k_dim: int,
    hidden_dim: int,
    num_experts: int,
    active_experts: int,
    expert_capacity: int,
    ff_bias: bool = False,
    multi_device: bool = True,
    attn_bias: bool = False,
    attn_dropout: float = 0.1,
    attn_resid_dropout: float = 0.0,
    name: str = None,
  ) -> None:
    super().__init__(name)
    self.pre_layer_norm = RMSNorm()
    self.post_attn_norm = RMSNorm()
    self.emd_dim = emd_dim

    self.attn = MultiHeadAttention(
      num_q_heads, 
      num_kv_heads,
      emd_dim,
      v_dim,
      k_dim,
      bias=attn_bias,
      attn_dropout=attn_dropout,
      resid_dropout=attn_resid_dropout
    )

    self.num_experts = num_experts
    if num_experts > 1:
      self.moe = MoEBlock(
        emd_dim, 
        hidden_dim,
        num_experts,
        active_experts,
        expert_capacity,
        ff_bias,
        multi_device
      )
    else: 
      self.ff = DenseFF(emd_dim, hidden_dim, bias=ff_bias)

  def __call__(self, x: jax.Array, kv_cache: jax.Array = None, is_training=True) -> jax.Array:
    """      
      B: batch size
      t: length of the input sequence
      D: embedding dimension
      K=Q: key/query dimension
      V: value dimension
      H: number of key/value heads

      x: B x t x D
      kv_cache: B x T x H x (K+V)
    """

    # Normalize before attention
    h = self.pre_layer_norm(x)
    h, new_kv_cache =  self.attn(h, h, h, kv_cache, is_training)
    # Residual connection
    h = h + x 
    # Post residual normalization
    r = self.post_attn_norm(h)
    if self.num_experts > 1:
      r = self.moe(r)
    else:
      r = self.ff(r)

    return r, new_kv_cache
  

class Embedding(hk.Module):
  def __init__(self, 
              emd_dim: int, 
              n_vocab: int,
              name: str = None) -> None:
    super().__init__(name)
    self.emd_dim = emd_dim
    self.vocab_size = n_vocab

  def encode(self, x: jax.Array) -> jax.Array:
    embedding = hk.get_parameter('embedding', (self.vocab_size, self.emd_dim), init=w_init)
    return jnp.einsum('ve, ...v -> ...e', embedding, x)

  def decode(self, x: jax.Array) -> jax.Array:
    embedding = hk.get_parameter('embedding', (self.vocab_size, self.emd_dim), init=w_init)
    return jnp.einsum('ve, ...e -> ...v', embedding, x)


class MoeTransformer(hk.Module):
    def __init__(self,
      depth: int, 
      n_vocab: int,
      emd_dim: int,
      num_q_heads, 
      num_kv_heads,
      v_dim: int,
      k_dim: int,
      hidden_dim: int,
      num_experts: int,
      active_experts: int,
      expert_capacity: int,
      ff_bias: bool = False,
      multi_device: bool = True,
      attn_bias: bool = False,
      attn_dropout: float = 0.0,
      attn_resid_dropout: float = 0.0,
      name: str = None,
    ): 
      super().__init__(name)

      self.block_config = {
        'emd_dim': emd_dim,
        'num_q_heads': num_q_heads,
        'num_kv_heads': num_kv_heads,
        'v_dim': v_dim,
        'k_dim': k_dim,
        'hidden_dim': hidden_dim,
        'num_experts': num_experts,
        'active_experts': active_experts,
        'expert_capacity': expert_capacity,
        'ff_bias': ff_bias,
        'attn_bias': attn_bias,
        'attn_dropout': attn_dropout,
        'attn_resid_dropout': attn_resid_dropout,
      }
      self.embedding = Embedding(emd_dim, n_vocab)
      self.final_norm = RMSNorm(name='final_norm')

      self.num_experts = num_experts
      self.depth = depth
      self.emd_dim = emd_dim
      self.n_vocab = n_vocab
    
    def get_embedding(self, x):
      """
        Obtain transformed functions for embedding encoder and decoder that accept shared parameters
        We do this because there is no native support for tieing the encoder and decoder in Haiku.
      """
      def encode(x):
        emd = Embedding(self.emd_dim, self.n_vocab)
        return emd.encode(x)
      
      def decode(x):
          emd = Embedding(self.emd_dim, self.n_vocab)
          return emd.decode(x)
      
      encoder_init, encoder = hk.transform(encode)
      _, decoder = hk.transform(decode)
      embedding_params = hk.experimental.transparent_lift(encoder_init, allow_reuse=True)(hk.next_rng_key(), x)

      return encoder, decoder, embedding_params

    def __call__(self, x: jax.Array, kv_caches: Dict[int, jax.Array] = {}, is_training=True) -> jax.Array:
      """
        B: batch size
        t: length of the input sequence
        T: length of the attended sequence
        D: embedding dimension
        K=Q: key/query dimension
        V: value dimension
        H: number of key/value heads

        x: B x t x D
        kv_caches[i]: B x T x H x (K+V)
      """
      encode, decode, embedding_params = self.get_embedding(x)
      h = encode(embedding_params, hk.next_rng_key(), x)

      new_kv_caches = {}
      for i in range(self.depth):
        # PERFORMANCE: module construction needs to happen here and not constructor for performance with jit        
        block = TransformerBlock(**self.block_config, name='block_{}'.format(i))
        h, new_kv_cache = block(h, kv_caches.get(i, None), is_training)
        new_kv_caches[i] = new_kv_cache

      h = self.final_norm(h)
      r = decode(embedding_params, None, h)

      return r, new_kv_caches