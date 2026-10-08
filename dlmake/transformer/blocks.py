from typing import Optional
import torch
import torch.nn as nn
from dlmake import Config
from .attention import Attention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        config: Config,
        layer: int,
        ff: Optional[nn.Module] = None,
        shallow_ff: Optional[nn.Module] = None,
    ):
        super(DecoderBlock, self).__init__()
        self.use_shallow = config.feed_forward.shallow_ff is not None
        assert config.attention.attention_method is not None, "Something must have gone terribly wrong for this error to appear"
        assert config.feed_forward.ff is not None, "Something must have gone terribly wrong for this error to appear"

        self.prenorm = config.prenorm
        self.postnorm = config.postnorm
        if self.prenorm:
            self.prenorm1 = config.norm(config.embed_dim, bias=config.bias)
            self.prenorm2 = config.norm(config.embed_dim, bias=config.bias)
        if self.postnorm:
            self.postnorm1 = config.norm(config.embed_dim, bias=config.bias)
            self.postnorm2 = config.norm(config.embed_dim, bias=config.bias)

        self.attn: Attention = config.attention.attention_method(config=config)

        if not self.use_shallow or layer >= config.feed_forward.shallow_threshold:
            if ff is None:
                assert not isinstance(config.feed_forward.ff, str), "Expected custom definition for a Feed Forward network, but got None"
                self.mlp: nn.Module = config.feed_forward.ff(config=config)
            else:
                self.mlp = ff(config=config)
        else:
            if isinstance(config.feed_forward.shallow_ff, str):
                assert shallow_ff is not None, "Custom shallow FeedForward setting was given but no class was provided to the constructor."
                self.mlp = shallow_ff(config=config)
            else:
                assert config.feed_forward.shallow_ff is not None, "Something must have gone terribly wrong for this error to appear"
                self.mlp = config.feed_forward.shallow_ff(config=config, shallow=True)

    def forward(self, x: torch.Tensor, start_pos: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.prenorm:
            x = self.prenorm1(x)
        if self.postnorm:
            x = x + self.postnorm1(self.attn(x, start_pos, mask))
        else:
            x = x + self.attn(x, start_pos, mask)
        if self.prenorm:
            x = self.prenorm2(x)
        if self.postnorm:
            x = x + self.postnorm2(self.mlp(x))
        else:
            x = x + self.mlp(x)
        return x
