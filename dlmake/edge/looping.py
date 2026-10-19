from typing import Literal
from dlmake import Config
import torch.nn as nn
import torch


class OuroExitGate(nn.Module):
    def __init__(self, config: Config):
        super(OuroExitGate, self).__init__()

        self.l = nn.Linear(in_features=config.embed_dim, out_features=1, dtype=config.dtype, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.sigmoid(self.l(x))


class OuroLoss(nn.Module):
    def __init__(self, beta: float = 0.1, dist: Literal["uniform"] = "uniform") -> None:
        super(OuroLoss, self).__init__()

        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")
        self.beta = beta
        self.dist = dist

    def forward(self, pred: torch.Tensor, target: torch.Tensor, probs: torch.Tensor):
        if self.dist == "uniform":
            dist = torch.ones_like(probs[0]) / probs.size(0)
        else:
            raise ValueError(f"'{self.dist}' is not one of the available values for the prior distribution.")
        loss = torch.tensor(0., dtype=pred.dtype, device=pred.device)
        for y, u in zip(pred, probs):
            loss += self.ce((y * u).view(-1, y.shape[-1]), target.view(-1)) - self.beta * self.kl(torch.log(u + 1e-7), dist)
        return loss
