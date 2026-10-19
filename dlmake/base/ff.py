import math
from dlmake import Config
import torch.nn as nn
import torch

class GPTFeedForward(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super(GPTFeedForward, self).__init__()

        self.l1 = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.feed_forward.linear_scale * config.embed_dim,
            dtype=config.dtype,
            bias=config.bias
        )
        # if isinstance(config.nlinear_params, list):
        #     self.nlinear = config.nlinear(*config.nlinear_params)
        # else:
        #     self.nlinear = config.nlinear(**config.nlinear_params)
        self.nlinear = config.feed_forward.nlinear()
        self.l2 = nn.Linear(
            in_features=config.feed_forward.linear_scale * config.embed_dim,
            out_features=config.embed_dim,
            dtype=config.dtype,
            bias=config.bias
        )
        self.dropout = nn.Dropout(config.feed_forward.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.nlinear(x)
        x = self.l2(x)
        x = self.dropout(x)
        return x


class LlamaFeedForward(nn.Module):
    def __init__(self, config: Config, **kwargs):
        super(LlamaFeedForward, self).__init__()

        self.l1 = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.feed_forward.linear_scale * config.embed_dim,
            dtype=config.dtype,
            bias=config.bias
        )
        self.l2 = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.feed_forward.linear_scale * config.embed_dim,
            dtype=config.dtype,
            bias=config.bias
        )
        self.l3 = nn.Linear(
            in_features=config.feed_forward.linear_scale * config.embed_dim,
            out_features=config.embed_dim,
            dtype=config.dtype,
            bias=config.bias
        )
        # if isinstance(config.nlinear_params, list):
        #     self.nlinear = config.nlinear(*config.nlinear_params)
        # else:
        #     self.nlinear = config.nlinear(**config.nlinear_params)
        self.nlinear = config.feed_forward.nlinear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nlinear(self.l1(x)) * self.l2(x)
        return self.l3(x)


class MoEMLP(nn.Module):
    def __init__(self, config: Config, shallow: bool = False):
        super(MoEMLP, self).__init__()
        if shallow:
            settings = config.feed_forward.shallow_moe
        else:
            settings = config.feed_forward.moe

        self.embed_dim = config.embed_dim
        self.hidden_dim = settings.hidden_dim
        self.min_dim = settings.latent_dim if settings.latent_shared else config.embed_dim

        self.w1 = nn.Linear(
            in_features=self.min_dim,
            out_features=self.hidden_dim,
            bias=settings.bias,
            dtype=config.dtype
        )
        self.w2 = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.min_dim,
            bias=settings.bias,
            dtype=config.dtype
        )
        self.w3 = nn.Linear(
            in_features=self.min_dim,
            out_features=self.hidden_dim,
            bias=settings.bias,
            dtype=config.dtype
        )
        self.nlinear = settings.nlinear()

    def forward(self, x: torch.Tensor):
        return self.w2(self.nlinear(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, config: Config, shallow: bool = False):
        super(Gate, self).__init__()

        if shallow:
            settings = config.feed_forward.shallow_moe
        else:
            settings = config.feed_forward.moe

        self.experts_per_tok = settings.experts_per_tok
        self.score_function = settings.score_function
        self.router_scaling = settings.router_scaling

        self.linear = nn.Linear(
            in_features=config.embed_dim,
            out_features=settings.n_experts,
            bias=settings.bias,
            dtype=config.dtype
        )

    def forward(self, x: torch.Tensor):
        scores = self.linear(x)
        if self.score_function == "softmax":
            scores = torch.softmax(scores, dim=-1)
        else:
            scores = torch.sigmoid(scores)
        indices = torch.topk(scores, self.experts_per_tok, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.score_function == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.router_scaling
        return weights.type_as(x), indices


class MoE(nn.Module):
    def __init__(self, config: Config, shallow: bool = False):
        super(MoE, self).__init__()

        if shallow:
            settings = config.feed_forward.shallow_moe
        else:
            settings = config.feed_forward.moe
        self.embed_dim = config.embed_dim
        self.n_experts = settings.n_experts
        self.hidden_dim = settings.hidden_dim
        self.use_shared = settings.use_shared
        self.use_latent = settings.use_latent
        self.latent_shared = settings.latent_shared
        # self.n_shared_experts = config.feed_forward.moe.n_shared_experts
        if self.use_shared:
            # assert self.n_shared_experts > 0, "Cannot use shared experts with 0 shared experts per MoE layer."
            self.shared = MoEMLP(config, shallow)
        self.nlinear = settings.nlinear()
        if self.use_latent or self.latent_shared:
            self.latent_proj = nn.Parameter(
                torch.empty(config.embed_dim, settings.latent_dim, dtype=config.dtype)
            )
            nn.init.kaiming_normal_(self.latent_proj, a=math.sqrt(5))

        if self.use_latent:
            min_dim = settings.latent_dim
        else:
            min_dim = self.embed_dim

        self.gate = Gate(config, shallow)
        self.expert_w1 = nn.Parameter(
            torch.empty(self.n_experts, self.hidden_dim, min_dim, dtype=config.dtype)
        )
        self.expert_w2 = nn.Parameter(
            torch.empty(self.n_experts, min_dim, self.hidden_dim, dtype=config.dtype)
        )
        self.expert_w3 = nn.Parameter(
            torch.empty(self.n_experts, self.hidden_dim, min_dim, dtype=config.dtype)
        )
        nn.init.kaiming_normal_(self.expert_w1, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.expert_w2, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.expert_w3, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.view(-1, self.embed_dim)
        weights, indices = self.gate(x)

        if not self.use_latent and self.use_shared and not self.latent_shared:
            x_shared = x
        elif not self.use_latent and self.use_shared and self.latent_shared:
            x_shared = x @ self.latent_proj
        elif self.use_latent and not self.use_shared:
            x = x @ self.latent_proj
        elif self.use_latent and self.use_shared and not self.latent_shared:
            x_shared = x.clone()
            x = x @ self.latent_proj
        elif self.use_latent and self.use_shared and self.latent_shared:
            x = x @ self.latent_proj
            x_shared = x

        # if self.use_latent:
        #     t = x @ self.latent_proj
        # else:
        #     t = x
        # if self.use_shared and not self.latent_shared:
        #     x_shared = x.clone()
        # elif self.use_shared and self.latent_shared:
        #     x_shared = t
        # x = t

        unique_experts = torch.unique(indices)
        idxs = []
        inputs = []
        scores = []
        token_idxs = []
        for i in unique_experts:
            idx = (indices == i).nonzero(as_tuple=False)
            inp = x[idx[:, 0]]
            idxs.append(inp.size(0))
            inputs.append(inp)
            scores.append(weights[idx[:, 0], idx[:, 1]])
            token_idxs.append(idx[:, 0])
        expert_input = torch.vstack(inputs).to(x.device)
        offsets = torch.cumsum(torch.tensor(idxs), dim=-1, dtype=torch.int32).to(x.device)
        full_scores = torch.hstack(scores)
        token_idxs = torch.hstack(token_idxs)

        w1 = torch.nn.functional.grouped_mm(
            mat_a=expert_input,
            mat_b=self.expert_w1[unique_experts].transpose(-1, -2),
            offs=offsets
        )
        w3 = torch.nn.functional.grouped_mm(
            mat_a=expert_input,
            mat_b=self.expert_w3[unique_experts].transpose(-1, -2),
            offs=offsets
        )
        out = torch.nn.functional.grouped_mm(
            mat_a=self.nlinear(w1) * w3,
            mat_b=self.expert_w2[unique_experts].transpose(-1, -2),
            offs=offsets
        )
        out = out * full_scores.unsqueeze(1)
        y = torch.zeros_like(x)
        for i in range(x.size(0)):
        # for i in unique_experts:
            # idxs = torch.where(token_idxs == i)[0]
            idxs = (token_idxs == i).nonzero(as_tuple=False).view(-1)
            y[i] = torch.sum(out[idxs], dim=0)

        if not self.use_latent and not self.use_shared:
            return y.view(shape)
        elif not self.use_latent and self.use_shared and not self.latent_shared:
            z = self.shared(x_shared)  # type:ignore
            return (y + z).view(shape)
        elif not self.use_latent and (self.use_shared and self.latent_shared):
            z = self.shared(x_shared)  # type: ignore
            return (y + (z @ self.latent_proj.T)).view(shape)
        elif self.use_latent and not self.use_shared:
            return (y @ self.latent_proj.T).view(shape)
        elif self.use_latent and self.use_shared and not self.latent_shared:
            z = self.shared(x_shared)  # type: ignore
            return ((y @ self.latent_proj.T) + z).view(shape)
        elif self.use_latent and self.use_shared and self.latent_shared:
            z = self.shared(x_shared)  # type: ignore
            return ((y + z) @ self.latent_proj.T).view(shape)
        else:
            raise RuntimeError("Return condition for MoE not one of the supported ones")



mlps = [
    GPTFeedForward,
    LlamaFeedForward,
    MoE,
]

__all__ = ["GPTFeedForward", "LlamaFeedForward", "MoE"]
