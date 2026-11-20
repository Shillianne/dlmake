from importlib import import_module
from dataclasses import dataclass
from .moe import MoESettings
from typing import Optional
import torch.nn as nn
import copy


state_params = {
    "linear_scale": (int, 4),
    "nlinear": (nn.Module, nn.ReLU),
    "dropout": (float, 0.0),
    "ff": ((nn.Module, str),),
    "shallow_ff": ((None.__class__, nn.Module, str), None),
    "shallow_threshold": (int, 100),
    "bias": (bool, True),
    "moe": MoESettings,
    "shallow_moe": MoESettings,
}


@dataclass
class FeedForwardSettings:
    linear_scale: int
    nlinear: nn.Module
    dropout: float
    ff: nn.Module | str
    shallow_ff: Optional[nn.Module | str]
    shallow_threshold: int
    bias: bool
    moe: MoESettings
    shallow_moe: MoESettings

    @classmethod
    def from_dict(cls, **kwargs):
        from dlmake import GPTFeedForward
        from ..base.ff import mlps

        params = {
            "linear_scale": (int, 4),
            "nlinear": (nn.Module, nn.ReLU),
            "dropout": (float, 0.0),
            "ff": ((nn.Module, str), GPTFeedForward),
            "shallow_ff": ((None.__class__, nn.Module, str), None),
            "shallow_threshold": (int, 100),
            "bias": (bool, True),
            "moe": MoESettings,
            "shallow_moe": MoESettings,
        }

        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in FeedForwardSettings.")
        values = {}
        nn_module = import_module("torch.nn")
        base_module = import_module("dlmake.base")
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
                if param == "nlinear":
                    try:
                        values[param] = getattr(nn_module, value)
                    except Exception as e:
                        try:
                            values[param] = getattr(base_module, value)
                        except Exception as e:
                            raise e
                elif param == "ff":
                    if value in (names := [mlp.__name__ for mlp in mlps]):
                        values[param] = getattr(base_module, value)
                    elif value == "custom":
                        values[param] = "custom"
                    else:
                        raise TypeError(f"The requested Feed Forward network ({value}) is not one of the available types ({names}).")
                elif param == "shallow_ff":
                    if value in (names := [mlp.__name__ for mlp in mlps]):
                        values[param] = getattr(base_module, value)
                    elif value == "custom":
                        values[param] = "custom"
                    else:
                        raise TypeError(f"The requested Feed Forward network ({value}) is not one of the available types ({names}).")
                else:
                    kind = params[param][0]
                    assert issubclass(value.__class__, kind), f"Attribute '{param}' isn't of the expected type {kind}"
                    # assert isinstance(value, kind), f"Attribute '{param}' isn't of the expected type {kind}"
                    values[param] = value
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

    def state_dict(self):
        d = self.__dict__
        state = {}
        for k, v in d.items():
            if isinstance(state_params[k], tuple):
                state[k] = v
            else:
                state[k] = v.state_dict()
        return state

    @classmethod
    def load_state_dict(cls, state: dict):
        d = copy.deepcopy(state)
        for k, v in state.items():
            if isinstance(v, dict):
                d[k] = state_params[k].load_state_dict(v)
        return cls(**d)
