import math
from dlmake import Config
import torch.nn as nn
import torch

class GPTFeedForward(nn.Module):
    def __init__(self, config: Config):
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
    def __init__(self, config: Config):
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
    def __init__(self, config: Config):
        super(MoEMLP, self).__init__()
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.feed_forward.moe.hidden_dim

        self.w1 = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.embed_dim,
            bias=config.feed_forward.moe.bias
        )
        self.w2 = nn.Linear(
            in_features=self.embed_dim,
            out_features=self.hidden_dim,
            bias=config.feed_forward.moe.bias
        )
        self.w3 = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.embed_dim,
            bias=config.feed_forward.moe.bias
        )
        self.nlinear = config.feed_forward.moe.nlinear()

    def forward(self, x: torch.Tensor):
        return self.w2(self.nlinear(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, config: Config):
        super(Gate, self).__init__()

        self.experts_per_tok = config.feed_forward.moe.experts_per_tok
        self.score_function = config.feed_forward.moe.score_function
        self.router_scaling = config.feed_forward.moe.router_scaling

        self.linear = nn.Linear(
            in_features=config.embed_dim,
            out_features=config.feed_forward.moe.n_experts,
            bias=config.feed_forward.moe.bias
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
    def __init__(self, config: Config):
        super(MoE, self).__init__()

        self.embed_dim = config.embed_dim
        self.n_experts = config.feed_forward.moe.n_experts
        self.hidden_dim = config.feed_forward.moe.hidden_dim
        self.use_shared = config.feed_forward.moe.use_shared
        self.use_latent = config.feed_forward.moe.use_latent
        self.latent_shared = config.feed_forward.moe.latent_shared
        # self.n_shared_experts = config.feed_forward.moe.n_shared_experts
        if self.use_shared:
            # assert self.n_shared_experts > 0, "Cannot use shared experts with 0 shared experts per MoE layer."
            self.shared = MoEMLP(config)
        self.nlinear = config.feed_forward.moe.nlinear()
        if self.use_latent:
            self.latent_proj = nn.Parameter(
                torch.empty(config.embed_dim, config.feed_forward.moe.latent_dim)
            )
            nn.init.kaiming_normal_(self.latent_proj, a=math.sqrt(5))
            min_dim = config.feed_forward.moe.latent_dim
        else:
            min_dim = self.embed_dim

        self.gate = Gate(config)
        self.expert_w1 = nn.Parameter(
            torch.empty(self.n_experts, self.hidden_dim, min_dim)
        )
        self.expert_w2 = nn.Parameter(
            torch.empty(self.n_experts, min_dim, self.hidden_dim)
        )
        self.expert_w3 = nn.Parameter(
            torch.empty(self.n_experts, self.hidden_dim, min_dim)
        )
        nn.init.kaiming_normal_(self.expert_w1, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.expert_w2, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.expert_w3, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        shape = x.shape
        x = x.view(-1, self.embed_dim)
        weights, indices = self.gate(x)
        if self.use_latent:
            t = x @ self.latent_proj
            if self.use_shared and not self.latent_shared:
                x_shared = x
            else:
                x_shared = t
            x = t
        else:
            x_shared = x


        unique_experts = torch.unique(indices)
        idxs = []
        inputs = []
        scores = []
        token_idxs = []
        for i in unique_experts:
            idx = (indices == i).nonzeros(as_tuple=False)
            inp = x[idx[:, 0]]
            idxs.append(inp.size(0))
            inputs.append(inp)
            scores.append(weights[idx[:, 0], idx[:, 1]])
            token_idxs.append(idx[:, 0])
        expert_input = torch.vstack(inputs)
        offsets = torch.cumsum(torch.tensor(idxs), dim=-1, dtype=torch.int32)
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
        for i in unique_experts:
            idxs = torch.where(token_idxs == i)[0]
            y[i] = torch.sum(out[idxs], dim=0)
        if self.use_shared:
            z = self.shared(x_shared)
            return (y + z).view(shape)
        return (y @ self.latent_proj.T).view(shape)


mlps = [
    GPTFeedForward,
    LlamaFeedForward,
    MoE,
]

__all__ = ["GPTFeedForward", "LlamaFeedForward"]
