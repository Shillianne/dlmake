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

class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim * 3, bias=False)
        self.fc2 = nn.Linear(in_features=config.embed_dim, out_features=config.embed_dim * 3, bias=False)
        self.fc3 = nn.Linear(in_features=config.embed_dim * 3, out_features=config.embed_dim, bias=False)

    def forward(self, x: torch.Tensor):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = torch.nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

__all__ = ["MLP", "SiLU", "SwiGLU"]
