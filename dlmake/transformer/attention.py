from .rope import precompute_rope_params, compute_rope
from abc import ABC, abstractmethod
from typing import Optional
from dlmake import Config
import torch.nn as nn
import torch
import math


class Attention(nn.Module, ABC):
    def __init__(self):
        """
        Base class for all attention methods.
        """

        super(Attention, self).__init__()

        self._is_training = True

    @abstractmethod
    def forward(self, x: torch.Tensor, position: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def create_kv_cache(self):
        pass


class MultiHeadAttention(Attention):
    def __init__(self, config: Config):
        """
        Basic Multi-Head Attention with KV cache

        Arguments:
            - config (Config): The configuration of the current model.

        This class mimics the implementation of the original GPT2 transformer, but adds certain features to optimize it or make it interchangeable
        with the other attention classes. These additions include a KV cache, QK normalization, RoPE and flash attention, all accessible and 
        customiazble with `Config`.
        """

        super(MultiHeadAttention, self).__init__()

        self.embed_dim = config.embed_dim
        self.nhead = config.nhead
        self.dropout = config.attention.dropout
        self.flash = config.attention.flash
        self.max_batch_size = config.max_batch_size
        self.max_ctx_size = config.max_ctx_size
        self.use_rope = config.attention.use_rope

        self.attn = nn.Linear(
            in_features=config.embed_dim,
            out_features=3 * config.embed_dim,
            bias=config.bias,
            dtype=config.dtype
        )
        self.proj = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.embed_dim,
            bias=config.bias,
            dtype=config.dtype
        )
        self.attn_dropout = nn.Dropout(config.attention.dropout)
        self.proj_dropout = nn.Dropout(config.attention.dropout)
        if config.attention.use_qknorm:
            self.qk_norm = config.attention.qk_norm_type
            self.q_norm = self.qk_norm(self.embed_dim // self.nhead, eps=1e-6, dtype=config.dtype)
            self.k_norm = self.qk_norm(self.embed_dim // self.nhead, eps=1e-6, dtype=config.dtype)
        else:
            self.q_norm = self.k_norm = None

        if config.attention.use_rope:
            self.cos: torch.Tensor
            self.sin: torch.Tensor
            cos, sin = precompute_rope_params(head_dim=self.embed_dim // self.nhead, max_cxt=config.max_ctx_size)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
        if not self._is_training:
            self.kv_cache: torch.Tensor
            self.register_buffer("kv_cache", torch.zeros(2, config.max_batch_size, config.nhead, config.max_ctx_size, config.embed_dim // config.nhead),
                                 persistent=False)

    def forward(self, x: torch.Tensor, position: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward method of the MultiHeadAttention class.

        Arguments:
            - x (torch.Tensor): Tensor of embeddings for the class to perform attention on.
            - start_pos (int): During training, this value is 0. During inference, it's set to the number of tokens processed,
              so that the input in this mode can be just the last token generated.
            - mask (Optional[torch.Tensor]): An optional mask to be applied to the attention mechanism. If not passed, it uses
              the standard no-lookahead mask.
        """

        B, T, E = x.shape
        end_pos = start_pos + T
        q, k, v = self.attn(x).split(self.embed_dim, dim=2)

        q: torch.Tensor = q.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)
        k: torch.Tensor = k.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)
        v: torch.Tensor = v.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        if self.use_rope:
            q = compute_rope(q, self.cos, self.sin, position)
            k = compute_rope(k, self.cos, self.sin, position)

        if not self._is_training:
            self.kv_cache[0, :B, :, start_pos:end_pos] = k
            self.kv_cache[1, :B, :, start_pos:end_pos] = v
            k = self.kv_cache[0, :B, :, :end_pos]
            v = self.kv_cache[1, :B, :, :end_pos]

        if self.flash and mask is None:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        elif self.flash and mask is not None:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
        else:
            att: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                att += mask.unsqueeze(-3)
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.proj_dropout(self.proj(out))
        return out

    def create_kv_cache(self):
        assert getattr(self, "kv_cache", None) is None, "A KV Cache is already defined for this model, please check if the model hasn't already been put on inference mode."
        if self._is_training:
            raise RuntimeError("A KV Cache cannot exist during training for performance reasons.")
        self.register_buffer("kv_cache", torch.zeros(2, self.max_batch_size, self.nhead, self.max_ctx_size, self.embed_dim // self.nhead),
                             persistent=False)


class MultiQueryAttention(Attention):
    def __init__(self, config: Config) -> None:
        """
        Multi Query Attention class

        Arguments:
            - config (Config): The configuration of the current model.

        This class modifies the base Multi-Head Attention class by joining the query, key and value projected inputs into a single
        head, reducing computational costs but reducing model quality in return. Besides this change, this class contains all the
        same modifications as MHA (KV cache, QK norm, RoPE, flash attention).
        """

        super(MultiQueryAttention, self).__init__()
        assert config.embed_dim % config.nhead == 0, "embedding dimension must be divisible by the number of attention heads"

        self.embed_dim = config.embed_dim
        self.nhead = config.nhead
        self.head_dim = config.embed_dim // config.nhead
        self.flash = config.attention.flash
        self.dropout = config.attention.dropout
        self.max_batch_size = config.max_batch_size
        self.max_ctx_size = config.max_ctx_size
        self.use_rope = config.attention.use_rope

        self.W_key = nn.Linear(
            in_features=config.embed_dim,
            out_features=self.head_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )
        self.W_value = nn.Linear(
            in_features=config.embed_dim,
            out_features=self.head_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )
        self.W_query = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.embed_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )
        self.attn_dropout = nn.Dropout(config.attention.dropout)
        self.out_dropout = nn.Dropout(config.attention.dropout)
        self.out_proj = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.embed_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )

        if config.attention.use_qknorm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6, dtype=config.dtype)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6, dtype=config.dtype)
        else:
            self.q_norm = self.k_norm = None

        if config.attention.use_rope:
            self.cos: torch.Tensor
            self.sin: torch.Tensor
            cos, sin = precompute_rope_params(head_dim=self.head_dim, max_cxt=config.max_ctx_size)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)

        if not self._is_training:
            self.kv_cache: torch.Tensor
            self.register_buffer("kv_cache", torch.zeros(2, config.max_batch_size, 1, config.max_ctx_size, self.head_dim),
                                 persistent=False)

    def forward(self, x: torch.Tensor, position: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, E = x.shape
        end_pos = start_pos + T

        q: torch.Tensor = self.W_query(x)
        k: torch.Tensor = self.W_key(x)
        v: torch.Tensor = self.W_value(x)

        q = q.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)
        k = k.view(B, T, 1, self.head_dim).transpose(1, 2)
        v = v.view(B, T, 1, self.head_dim).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        if self.use_rope:
            q = compute_rope(q, self.cos, self.sin, position)
            k = compute_rope(k, self.cos, self.sin, position)

        if not self._is_training:
            self.kv_cache[0, :B, :, start_pos:end_pos] = k
            self.kv_cache[1, :B, :, start_pos:end_pos] = v
            k = self.kv_cache[0, :B, :, :end_pos].repeat_interleave(self.nhead, dim=1)
            v = self.kv_cache[1, :B, :, :end_pos].repeat_interleave(self.nhead, dim=1)
        else:
            # The resulting tensors are not contiguous
            k = k.expand(-1, self.nhead, -1, -1)
            v = v.expand(-1, self.nhead, -1, -1)

        if self.flash and mask is None:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        elif self.flash and mask is not None:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
        else:
            att: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1) ** 0.5)
            if mask is not None:
                att += mask.unsqueeze(-3)
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.out_dropout(self.out_proj(out))
        return out

    def create_kv_cache(self):
        assert getattr(self, "kv_cache", None) is None, "A KV Cache is already defined for this model, please check if the model hasn't already been put on inference mode."
        if self._is_training:
            raise RuntimeError("A KV Cache cannot exist during training for performance reasons.")
        self.register_buffer("kv_cache", torch.zeros(2, self.max_batch_size, 1, self.max_ctx_size, self.head_dim),
                             persistent=False)


class GroupedQueryAttention(Attention):
    def __init__(self, config: Config) -> None:
        """
        Implementation of Grouped Query Attetion.

        Arguments:
            - config (Config): The configuration of the current model.

        This class adapts Multi Query Attention's compromising approach to the performance of the model by setting a number of groups of attention heads,
        all heads within a group sharing the same query, key and value tensors. This consumes more memory and compute than MQA, but it improves model performance.
        In practice, it's barely worse than MHA while improving memory usage and inference speed.
        """
        super(GroupedQueryAttention, self).__init__()
        assert config.embed_dim % config.nhead == 0, "embedding dimension must be divisible by the number of attention heads"
        assert config.attention.n_kv_groups is not None, "A number of kv groups is required for Grouped Query Attention"
        assert config.nhead % config.attention.n_kv_groups == 0, "number of attention heads must be divisible by the number of key-value groups"

        self.embed_dim = config.embed_dim
        self.nhead = config.nhead
        self.head_dim = config.embed_dim // config.nhead
        self.flash = config.attention.flash
        self.dropout = config.attention.dropout
        self.max_batch_size = config.max_batch_size
        self.max_ctx_size = config.max_ctx_size
        self.use_rope = config.attention.use_rope

        self.W_key = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.attention.n_kv_groups * self.head_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )
        self.W_value = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.attention.n_kv_groups * self.head_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )
        self.n_kv_groups = config.attention.n_kv_groups
        self.group_size = config.nhead // config.attention.n_kv_groups

        self.W_query = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.embed_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )
        self.attn_dropout = nn.Dropout(config.attention.dropout)
        self.out_dropout = nn.Dropout(config.attention.dropout)
        self.out_proj = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.embed_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )

        if config.attention.use_qknorm:
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6, dtype=config.dtype)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6, dtype=config.dtype)
        else:
            self.q_norm = self.k_norm = None

        if config.attention.use_rope:
            self.cos: torch.Tensor
            self.sin: torch.Tensor
            cos, sin = precompute_rope_params(head_dim=self.head_dim, max_cxt=config.max_ctx_size)
            self.register_buffer("cos",  cos)
            self.register_buffer("sin", sin)

        if not self._is_training:
            self.kv_cache: torch.Tensor
            self.register_buffer("kv_cache", torch.zeros(2, config.max_batch_size, self.n_kv_groups, config.max_ctx_size, self.head_dim),
                                 persistent=False)

    def forward(self, x: torch.Tensor, position: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, E = x.shape
        end_pos = start_pos + T

        q: torch.Tensor = self.W_query(x)
        k: torch.Tensor = self.W_key(x)
        v: torch.Tensor = self.W_value(x)

        q = q.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)  # shape: (batch, heads, ctx, head_dim)
        q = q.view(B, self.n_kv_groups, self.group_size, T, E // self.nhead)  # shape: (batch, kv_group, group_size, ctx, head_dim)
        k = k.view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)  # shape: (batch, kv_group, ctx, head_dim)
        v = v.view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)  # shape: (batch, kv_group, ctx, head_dim)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        if self.use_rope:
            q = compute_rope(q, self.cos, self.sin, position)
            k = compute_rope(k, self.cos, self.sin, position)

        if not self._is_training:
            self.kv_cache[0, :B, :, start_pos:end_pos] = k
            self.kv_cache[1, :B, :, start_pos:end_pos] = v
            k = self.kv_cache[0, :B, :, :end_pos].unsqueeze(2).expand(-1, self.n_kv_groups, self.group_size, -1, -1)
            v = self.kv_cache[1, :B, :, :end_pos].unsqueeze(2).expand(-1, self.n_kv_groups, self.group_size, -1, -1)
        else:
            # The resulting tensors are not contiguous
            k = k.unsqueeze(2).expand(-1, self.n_kv_groups, self.group_size, -1, -1)  # shape: (batch, kv_group, group_size, ctx, head_dim)
            v = v.unsqueeze(2).expand(-1, self.n_kv_groups, self.group_size, -1, -1)  # shape: (batch, kv_group, group_size, ctx, head_dim)
            # k = k.expand(-1, self.group_size, -1, -1)  # shape: (batch, heads, ctx, head_dim)
            # v = v.expand(-1, self.group_size, -1, -1)  # shape: (batch, heads, ctx, head_dim)

        if self.flash and mask is None:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        elif self.flash and mask is not None:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0)
        else:
            att: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1) ** 0.5)
            if mask is not None:
                att += mask.unsqueeze(-3)
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v

        out = out.view(B, self.nhead, T, E // self.nhead).transpose(1, 2).contiguous().view(B, T, E)
        out = self.out_dropout(self.out_proj(out))
        return out

    def create_kv_cache(self):
        assert getattr(self, "kv_cache", None) is None, "A KV Cache is already defined for this model, please check if the model hasn't already been put on inference mode."
        if self._is_training:
            raise RuntimeError("A KV Cache cannot exist during training for performance reasons.")
        self.register_buffer("kv_cache", torch.zeros(2, self.max_batch_size, self.n_kv_groups, self.max_ctx_size, self.head_dim),
                             persistent=False)


class MultiHeadLatentAttention(Attention):
    def __init__(self, config: Config):
        """
        Implementation of Multi-Head Latent Attention from the Deepseek team.

        Arguments:
            - config (Config): The configuration of the current model

        This attention method compreses the key and value tensors into a latent space that also serves as the KV cache space. By allowing the model
        to learn how to best project into the inputs into this latent space and by exploiting some algebraic properties, MLA obtains better memory usage
        and inference speed while improving model performance with respect to MHA. However, external studies have shown that these gains only start to show
        in larger models, so it's recommended to use GQA instead of this attention for smaller models.
        """

        super(MultiHeadLatentAttention, self).__init__()
        assert config.attention.kv_lora_dim is not None and config.attention.q_head_dim is not None and config.attention.v_head_dim is not None

        self.embed_dim = config.embed_dim
        self.nheads = config.nhead
        self.q_head_dim = config.attention.q_head_dim
        self.kv_lora_dim = config.attention.kv_lora_dim
        self.v_head_dim = config.attention.v_head_dim
        self.max_batch_size = config.max_batch_size
        self.max_ctx_size = config.max_ctx_size

        self.wq = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.nhead * config.attention.q_head_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )
        self.wkv_a = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.attention.kv_lora_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )
        self.kv_norm = nn.RMSNorm(config.attention.kv_lora_dim, dtype=config.dtype)
        self.wkv_b = nn.Linear(
            in_features=config.attention.kv_lora_dim,
            out_features=config.nhead * (config.attention.q_head_dim + config.attention.v_head_dim),
            bias=config.attention.bias,
            dtype=config.dtype
        )
        self.wo = nn.Linear(
            in_features=config.nhead * config.attention.v_head_dim,
            out_features=config.embed_dim,
            bias=config.attention.bias,
            dtype=config.dtype
        )

        if not self._is_training:
            self.kv_cache: torch.Tensor
            self.register_buffer("kv_cache", torch.zeros(config.max_batch_size, config.max_ctx_size, config.attention.kv_lora_dim), persistent=False)

    def forward(self, x: torch.Tensor, position: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        end_pos = start_pos + T

        q: torch.Tensor = self.wq(x)
        q = q.view(B, T, self.nheads, self.q_head_dim)
        kv: torch.Tensor = self.wkv_a(x)
        wkv_b = self.wkv_b.weight  # (nheads * (q_head_dim + v_head_dim), kv_lora_dim)
        wkv_b = wkv_b.view(self.nheads, (self.q_head_dim + self.v_head_dim), self.kv_lora_dim)
        q = torch.einsum("bshq,hqc->bshc", q, wkv_b[:, :self.q_head_dim])  # (batch, t, nheads, kv_lora_dim)
        if not self._is_training:
            self.kv_cache[:B, start_pos:end_pos] = kv
        scores = torch.einsum("bshc,btc->bsht", q, kv if self._is_training else self.kv_cache[:B, :end_pos]) * (self.q_head_dim ** -0.5)
        if mask is not None:
            scores += mask.unsqueeze(-2)
        scores = torch.softmax(scores, dim=-1).type_as(x)
        x = torch.einsum("bsht,btc->bshc", scores, kv if self._is_training else self.kv_cache[:B, :end_pos])
        x = torch.einsum("bshc,hvc->bshv", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x

    def create_kv_cache(self):
        assert getattr(self, "kv_cache", None) is None, "A KV Cache is already defined for this model, please check if the model hasn't already been put on inference mode."
        if self._is_training:
            raise RuntimeError("A KV Cache cannot exist during training for performance reasons.")
        self.register_buffer("kv_cache", torch.zeros(self.max_batch_size, self.max_ctx_size, self.kv_lora_dim), persistent=False)


attention_types = [
    MultiHeadAttention,
    MultiQueryAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention
]
__all__ = ["MultiHeadAttention", "MultiQueryAttention", "GroupedQueryAttention", "MultiHeadLatentAttention"]
