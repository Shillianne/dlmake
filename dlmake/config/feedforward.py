from dataclasses import dataclass
from typing import Optional
from .moe import MoESettings
import torch.nn as nn


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
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
                if param == "nlinear":
                    try:
                        exec(f"nn.{value}")
                    except Exception as e:
                        try:
                            exec(f"from ..base import {value}")
                        except Exception as e:
                            raise e
                        else:
                            exec(f"values['{param}'] = {value}")
                    else:
                        exec(f"values['{param}'] = nn.{value}")
                elif param == "ff":
                    if value in (names := [mlp.__name__ for mlp in mlps]):
                        exec(f"from ..base.ff import {value}")
                        exec(f"values['{param}'] = {value}")
                    elif value == "custom":
                        values[param] = "custom"
                    else:
                        raise TypeError(f"The requested Feed Forward network ({value}) is not one of the available types ({names}).")
                elif param == "shallow_ff":
                    if value in (names := [mlp.__name__ for mlp in mlps]):
                        exec(f"from ..base.ff import {value}")
                        exec(f"values['{param}'] = {value}")
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
