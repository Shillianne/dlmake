from typing import Optional
import torch
import torch.nn as nn
from dlmake import Config
from .attention import Attention


class DecoderBlock(nn.Module):
    def __init__(self, config: Config, ff: Optional[nn.Module] = None):
        super(DecoderBlock, self).__init__()
        assert config.attention_method is not None, "Something must have gone terribly wrong for this error to appear"
        assert config.ff is not None, "Something must have gone terribly wrong for this error to appear"
        self.norm1 = config.norm(config.embed_dim, bias=config.bias)
        # self.ln1 = nn.LayerNorm(config.embed_dim, bias=config.bias)
        self.attn: Attention = config.attention_method(config=config)
        # self.ln2 = nn.LayerNorm(config.embed_dim, bias=config.bias)
        self.norm2 = config.norm(config.embed_dim, bias=config.bias)
        if ff is None:
            assert not isinstance(config.ff, str), "Expected custom definition for a Feed Forward network, but got None"
            self.mlp: nn.Module = config.ff(config=config)
        else:
            self.mlp = ff(config=config)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        l = self.norm1(x)
        x = x + self.attn(l, start_pos, mask)
        x = x + self.mlp(self.norm2(x))
        return x
