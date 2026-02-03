from typing import Callable, Optional
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from dlmake import Config
import math


class Attention(nn.Module, ABC):
    def __init__(self):
        super(Attention, self).__init__()

        self._is_training = True

    @abstractmethod
    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def create_kv_cache(self):
        pass


class MultiHeadAttention(Attention):
    def __init__(self, config: Config):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = config.embed_dim
        self.nhead = config.nhead
        self.dropout = config.dropout_prob
        self.flash = config.flash
        self.max_batch_size = config.max_batch_size
        self.max_ctx_size = config.max_ctx_size

        self.attn = nn.Linear(in_features=config.embed_dim, out_features=3 * config.embed_dim, bias=config.bias)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout_prob)
        self.proj_dropout = nn.Dropout(config.dropout_prob)
        if not self._is_training:
            self.kv_cache: torch.Tensor
            self.register_buffer("kv_cache", torch.zeros(2, config.max_batch_size, config.nhead, config.max_ctx_size, config.embed_dim // config.nhead),
                                 persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        super(MultiQueryAttention, self).__init__()
        assert config.embed_dim % config.nhead == 0, "embedding dimension must be divisible by the number of attention heads"

        self.embed_dim = config.embed_dim
        self.nhead = config.nhead
        self.head_dim = config.embed_dim // config.nhead
        self.flash = config.flash
        self.dropout = config.dropout_prob
        self.max_batch_size = config.max_batch_size
        self.max_ctx_size = config.max_ctx_size

        self.W_key = nn.Linear(config.embed_dim, self.head_dim, bias=False)
        self.W_value = nn.Linear(config.embed_dim, self.head_dim, bias=False)
        self.W_query = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout_prob)
        self.out_dropout = nn.Dropout(config.dropout_prob)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        if not self._is_training:
            self.register_buffer("kv_cache", torch.zeros(2, config.max_batch_size, 1, config.max_ctx_size, self.head_dim),
                                 persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, E = x.shape
        end_pos = start_pos + T

        q: torch.Tensor = self.W_query(x)
        k: torch.Tensor = self.W_key(x)
        v: torch.Tensor = self.W_value(x)

        q = q.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)
        k = k.view(B, T, 1, self.head_dim).transpose(1, 2)
        v = v.view(B, T, 1, self.head_dim).transpose(1, 2)
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
        super(GroupedQueryAttention, self).__init__()
        assert config.embed_dim % config.nhead == 0, "embedding dimension must be divisible by the number of attention heads"
        assert config.n_kv_groups is not None, "A number of kv groups is required for Grouped Query Attention"
        assert config.nhead % config.n_kv_groups == 0, "number of attention heads must be divisible by the number of key-value groups"

        self.embed_dim = config.embed_dim
        self.nhead = config.nhead
        self.head_dim = config.embed_dim // config.nhead
        self.flash = config.flash
        self.dropout = config.dropout_prob
        self.max_batch_size = config.max_batch_size
        self.max_ctx_size = config.max_ctx_size

        self.W_key = nn.Linear(config.embed_dim, config.n_kv_groups * self.head_dim, bias=False)
        self.W_value = nn.Linear(config.embed_dim, config.n_kv_groups * self.head_dim, bias=False)
        self.n_kv_groups = config.n_kv_groups
        self.group_size = config.nhead // config.n_kv_groups

        self.W_query = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout_prob)
        self.out_dropout = nn.Dropout(config.dropout_prob)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim, bias=False)

        if not self._is_training:
            self.register_buffer("kv_cache", torch.zeros(2, config.max_batch_size, self.n_kv_groups, config.max_ctx_size, self.head_dim),
                                 persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, E = x.shape
        end_pos = start_pos + T

        q: torch.Tensor = self.W_query(x)
        k: torch.Tensor = self.W_key(x)
        v: torch.Tensor = self.W_value(x)

        q = q.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)  # shape: (batch, heads, ctx, head_dim)
        k = k.view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)  # shape: (batch, kv_group, ctx, head_dim)
        v = v.view(B, T, self.n_kv_groups, self.head_dim).transpose(1, 2)  # shape: (batch, kv_group, ctx, head_dim)
        if not self._is_training:
            self.kv_cache[0, :B, :, start_pos:end_pos] = k
            self.kv_cache[1, :B, :, start_pos:end_pos] = v
            k = self.kv_cache[0, :B, :, :end_pos].repeat_interleave(self.group_size, dim=1)
            v = self.kv_cache[1, :B, :, :end_pos].repeat_interleave(self.group_size, dim=1)
        else:
            # The resulting tensors are not contiguous
            k = k.expand(-1, self.group_size, -1, -1)  # shape: (batch, heads, ctx, head_dim)
            v = v.expand(-1, self.group_size, -1, -1)  # shape: (batch, heads, ctx, head_dim)

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
        self.register_buffer("kv_cache", torch.zeros(2, self.max_batch_size, self.n_kv_groups, self.max_ctx_size, self.head_dim),
                             persistent=False)


class MultiHeadLatentAttention(Attention):
    def __init__(self, config: Config):
        super(MultiHeadLatentAttention, self).__init__()
        assert config.kv_lora_dim is not None and config.q_head_dim is not None and config.v_head_dim is not None

        self.embed_dim = config.embed_dim
        self.nheads = config.nhead
        self.q_head_dim = config.q_head_dim
        self.kv_lora_dim = config.kv_lora_dim
        self.v_head_dim = config.v_head_dim
        self.max_batch_size = config.max_batch_size
        self.max_ctx_size = config.max_ctx_size

        self.wq = nn.Linear(in_features=config.embed_dim, out_features=config.nhead * config.q_head_dim, bias=False)
        self.wkv_a = nn.Linear(in_features=config.embed_dim, out_features=config.kv_lora_dim, bias=False)
        self.kv_norm = nn.RMSNorm(config.kv_lora_dim)
        self.wkv_b = nn.Linear(in_features=config.kv_lora_dim, out_features=config.nhead * (config.q_head_dim + config.v_head_dim), bias=False)
        self.wo = nn.Linear(in_features=config.nhead * config.v_head_dim, out_features=config.embed_dim, bias=False)

        if not self._is_training:
            self.register_buffer("kv_cache", torch.zeros(config.max_batch_size, config.max_ctx_size, config.kv_lora_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
