import torch
import torch.nn as nn
from dlmake import Config

class MLP(nn.Module):
    def __init__(self, config: Config):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(in_features=config.embed_dim, out_features=4 * config.embed_dim, bias=config.bias)
        if isinstance(config.nlinear_params, list):
            self.nlinear = config.nlinear(*config.nlinear_params)
        else:
            self.nlinear = config.nlinear(**config.nlinear_params)
        self.l2 = nn.Linear(in_features=4 * config.embed_dim, out_features=config.embed_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.nlinear(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x

__all__ = ["MLP"]
