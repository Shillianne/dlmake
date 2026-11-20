from dlmake import Config
import torch.nn as nn
import torch


class PrevContextCompression(nn.Module):
    def __init__(self, config: Config):
        super(PrevContextCompression, self).__init__()
        assert config.ctx_size % config.cc.factor == 0, "Context size has to be divisible by compression factor"

        self.ctx = config.ctx_size
        self.factor = config.cc.factor
        self.embed = config.embed_dim
        self.l1 = nn.Linear(
            in_features=config.embed_dim,
            out_features=self.ctx // self.factor,
            bias=False,
            dtype=config.dtype,
        )
        self.l2 = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.embed_dim,
            bias=False,
            dtype=config.dtype,
        )

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.embed)
        scores = self.l1(x)
        scores = torch.softmax(scores, dim=-1)
        weights, indices = torch.topk(scores, k=1, dim=-1)
        unique_groups = torch.unique(indices)
        outs: list[torch.Tensor] = []
        for g in unique_groups:
            idxs = torch.where(indices == g)[0]
            outs.append(torch.sum(self.l2(x[idxs]) * weights[idxs]))
        return torch.stack(outs, dim=0)


class ContextCompression(nn.Module):
    def __init__(self, config: Config):
        super(ContextCompression, self).__init__()
        assert config.ctx_size % config.cc.factor == 0, "Context size has to be divisible by compression factor"

        self.l = nn.Linear(
            in_features=config.ctx_size,
            out_features=config.ctx_size // config.cc.factor,
            bias=False,
            dtype=config.dtype,
        )

    def forward(self, x: torch.Tensor):
        x = self.l(x.transpose(1, 2))
        return x.transpose(1, 2).contiguous()

