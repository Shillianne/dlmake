from dlmake import Config
import torch.nn as nn
import torch


class OuroExitGate(nn.Module):
    def __init__(self, config: Config):
        super(OuroExitGate, self).__init__()

        self.l = nn.Linear(in_features=config.embed_dim, out_features=1, bias=True)

    def forward(self, x: torch.Tensor):
        return nn.functional.sigmoid(self.l(x))
