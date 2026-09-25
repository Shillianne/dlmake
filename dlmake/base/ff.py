from dlmake import Config
import torch.nn as nn
import torch

class GPTFeedForward(nn.Module):
    def __init__(self, config: Config):
        super(GPTFeedForward, self).__init__()

        self.l1 = nn.Linear(in_features=config.embed_dim, out_features=config.linear_scale * config.embed_dim, dtype=config.dtype, bias=config.bias)
        if isinstance(config.nlinear_params, list):
            self.nlinear = config.nlinear(*config.nlinear_params)
        else:
            self.nlinear = config.nlinear(**config.nlinear_params)
        self.l2 = nn.Linear(in_features=config.linear_scale * config.embed_dim, out_features=config.embed_dim, dtype=config.dtype, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.nlinear(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x


class LlamaFeedForward(nn.Module):
    def __init__(self, config: Config):
        super(LlamaFeedForward, self).__init__()

        self.l1 = nn.Linear(in_features=config.embed_dim, out_features=config.linear_scale * config.embed_dim, dtype=config.dtype, bias=config.bias)
        self.l2 = nn.Linear(in_features=config.embed_dim, out_features=config.linear_scale * config.embed_dim, dtype=config.dtype, bias=config.bias)
        self.l3 = nn.Linear(in_features=config.linear_scale * config.embed_dim, out_features=config.embed_dim, dtype=config.dtype, bias=config.bias)
        if isinstance(config.nlinear_params, list):
            self.nlinear = config.nlinear(*config.nlinear_params)
        else:
            self.nlinear = config.nlinear(**config.nlinear_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nlinear(self.l1(x)) * self.l2(x)
        return self.l3(x)


mlps = [
    GPTFeedForward,
    LlamaFeedForward,
]

__all__ = ["GPTFeedForward", "LlamaFeedForward"]
