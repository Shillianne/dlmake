from typing import Literal, get_args
from dataclasses import dataclass
from torch import nn


params = {
    "n_experts": (int, 1),
    "experts_per_tok": (int, 1),
    "use_shared": (bool, False),
    # "n_shared_experts": (int, 1),
    "hidden_dim": (int, 128),
    "nlinear": (nn.Module, nn.SiLU),
    "router_scaling": (float, 1.0),
    "score_function": (Literal["softmax", "sigmoid"], "softmax"),
    "bias": (bool, True),
    "use_latent": (bool, False),
    "latent_dim": (int, 1),
    "latent_shared": (bool, False),
}

@dataclass
class MoESettings:
    n_experts: int
    experts_per_tok: int
    use_shared: bool
    # n_shared_experts: int
    hidden_dim: int
    nlinear: nn.Module
    router_scaling: float
    score_function: Literal["softmax", "sigmoid"]
    bias: bool
    use_latent: bool
    latent_dim: int
    latent_shared: bool

    @classmethod
    def from_dict(cls, **kwargs):
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in MoESettings.")
        values = {}
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
                if param == "nlinear":
                    try:
                        exec(f"nn.{value}")
                    except Exception as e:
                        raise e
                    else:
                        exec(f"values['{param}'] = nn.{value}")
                elif param == "score_function":
                    if value not in get_args(params[param][0]):
                        raise ValueError(f"Attribute 'score_function' does not support a value of '{value}'.")
                    values[param] = value
                else:
                    kind = params[param][0]
                    # assert isinstance(kwargs[param], kind), f"Attribute '{param}' isn't of the expected type {kind}"
                    assert issubclass(value.__class__, kind), f"Attribute '{param}' isn't of the expected type {kind}"
                    values[param] = kwargs[param]
            else:
                values[param] = params[param].from_dict(**kwargs[param])
        remaining = set(params.keys()).difference(set(values.keys()))
        for param in list(remaining):
            if isinstance(params[param], tuple):
                default = params[param][1]
                values[param] = default
            else:
                values[param] = params[param].from_dict()

        return cls(**values)
