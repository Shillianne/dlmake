import torch
import torch.nn as nn
from dlmake import Config
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = config.embed_dim
        self.nhead = config.nhead
        self.dropout = config.dropout_prob
        self.flash = config.flash

        self.attn = nn.Linear(in_features=config.embed_dim, out_features=3 * config.embed_dim, bias=config.bias)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout_prob)
        self.proj_dropout = nn.Dropout(config.dropout_prob)
        if not config.flash:
            print("Not using Flash Attention")
            self.register_buffer("bias", torch.tril(torch.ones(config.ctx_size, config.ctx_size)).view(1, 1, config.ctx_size, config.ctx_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E = x.shape
        q, k, v = self.attn(x).split(self.embed_dim, dim=2)

        q = q.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)
        k = k.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)
        v = v.view(B, T, self.nhead, E // self.nhead).transpose(1, 2)

        if self.flash:
            out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att: torch.Tensor = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('inf'))
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.proj_dropout(self.proj(out))
        return out

