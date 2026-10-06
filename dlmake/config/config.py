from .feedforward import FeedForwardSettings
from .attention import AttentionSettings
from .training import TrainingSettings
from dataclasses import dataclass
from yaml import load, Loader, loader
import torch.nn as nn
import torch
import json


params = {
    "name": (str, "model"),
    "ctx_size": (int, 64),
    "max_ctx_size": (int, 64),
    "vocab_size": (int, 256),
    "embed_dim": (int, 1),
    "nhead": (int, 1),
    "nlayers": (int, 1),
    "batch_size": (int, 1),
    "max_batch_size": (int, 1),
    "bias": (bool, True),
    "dtype": (torch.dtype, torch.bfloat16),
    "norm": (nn.Module, nn.LayerNorm),
    "prenorm": (bool, True),
    "postnorm": (bool, False),
    "training": TrainingSettings,
    "feed_forward": FeedForwardSettings,
    "attention": AttentionSettings,
}


@dataclass
class Config:
    name: str
    ctx_size: int
    max_ctx_size: int
    vocab_size: int
    embed_dim: int
    nhead: int
    nlayers: int
    batch_size: int
    max_batch_size: int
    bias: bool
    dtype: torch.dtype
    norm: nn.Module
    prenorm: bool
    postnorm: bool
    use_looping: bool
    training: TrainingSettings
    feed_forward: FeedForwardSettings
    attention: AttentionSettings

    @classmethod
    def from_json(cls, filename: str):
        with open(filename, "r") as f:
            kwargs = json.load(f)
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in LoggingSettings.")
        values = {}
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
                if param == "dtype":
                    try:
                        exec(f"from torch import {value}")
                    except Exception as e:
                        raise e
                    else:
                        exec(f"values['{param}'] = {value}")
                elif param == "norm":
                    try:
                        exec(f"from torch.nn import {value}")
                    except Exception as e:
                        raise e
                    else:
                        exec(f"values['{param}'] = {value}")
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

    @classmethod
    def from_yaml(cls, filename: str):
        with open(filename, "r") as f:
            kwargs = load(f, Loader=Loader)
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in LoggingSettings.")
        values = {}
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
                if param == "dtype":
                    try:
                        exec(f"from torch import {value}")
                    except Exception as e:
                        raise e
                    else:
                        exec(f"values['{param}'] = {value}")
                elif param == "norm":
                    try:
                        exec(f"from torch.nn import {value}")
                    except Exception as e:
                        raise e
                    else:
                        exec(f"values['{param}'] = {value}")
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

        cls.check(values)
        return cls(**values)

    @staticmethod
    def check(values: dict):
        assert values["max_ctx_size"] >= values["ctx_size"], "Maximum ctx size has to be at least equal to the ctx size"
        assert values["max_batch_size"] >= values["batch_size"], "Maximum batch size has to be at least equal to the batch size"
        assert values["prenorm"] or values["postnorm"], "At least one of prenorm or postnorm have to be true"
