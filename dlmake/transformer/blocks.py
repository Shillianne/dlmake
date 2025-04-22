from typing import Optional
import torch
import torch.nn as nn
from dlmake import Config
from .attention import Attention
from dlmake.base import MLP

class DecoderBlock(nn.Module):
    def __init__(self, config: Config):
        super(DecoderBlock, self).__init__()
        assert config.attention_method is not None, "Something must have gone terribly wrong for this error to appear"
        self.ln1 = nn.LayerNorm(config.embed_dim, bias=config.bias)
        self.attn: Attention = config.attention_method(config=config)
        self.ln2 = nn.LayerNorm(config.embed_dim, bias=config.bias)
        self.mlp = MLP(config=config)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        l = self.ln1(x)
        x = x + self.attn(l, mask)
        x = x + self.mlp(self.ln2(x))
        return x
