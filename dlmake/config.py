from typing import Literal, Type, Union, List, Dict, Any, Optional
from torch.optim import AdamW
import torch.nn as nn
import json

Module = Type[nn.Module]

class Config:
    def __init__(self,
                 ctx_size: int = 64,
                 vocab_size: int = 256,
                 embed_dim: int = 1,
                 batch_size: int = 1,
                 nhead: int = 1,
                 nlayers: int = 1,
                 bias: bool = True,
                 nlinear: Module = nn.ReLU,
                 nlinear_params: Union[List[Any], Dict[str, Any]] = [],
                 dropout_prob: float = 0.0,
                 optimizer = AdamW,
                 flash: bool = False,
                 verbose: bool = True,
                 compile: bool = False,
                 checkpoint_rate: int = 10,
                 use_val: bool = False,
                 validation_rate: Optional[int] = None,
                 benchmark_rate: Optional[int] = None,
                 new: bool = True,
                 time_precision: Literal['s', 'ms'] = 'ms') -> None:
        self.ctx_size = ctx_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.nhead = nhead
        self.nlayers = nlayers
        self.bias = bias
        self.nlinear = nlinear
        self.nlinear_params = nlinear_params
        self.dropout_prob = dropout_prob
        self.optimizer = optimizer
        self.flash = flash
        self.verbose = verbose
        self.compile = compile
        self.checkpoint_rate = checkpoint_rate
        self.use_val = use_val
        self.validation_rate = validation_rate
        self.benchmark_rate = benchmark_rate
        self.new = new
        self.time_precision = time_precision

    @classmethod
    def from_json(cls, filename: str) -> "Config":
        with open(filename, "r") as f:
            config = json.load(f)
        params = ["ctx_size", "vocab_size", "embed_dim", "batch_size", "nhead", "nlayers",
                  "bias", "nlinear", "nlinear_params", "dropout_prob", "optimizer",
                  "flash", "verbose", "compile", "checkpoint_rate", "use_val", "validation_rate",
                  "benchmark_rate", "new", "time_precision"]
        defaults = [64, 256, 1, 1, 1, 1, True, nn.ReLU, [], 0.0, AdamW, False, True, False, 10, False, None, None, True, 'ms']
        defined = []
        values = {}
        for i, (param, value) in enumerate(config.items()):
            if param not in params:
                raise KeyError("Argument defined in JSON file not found in available arguments.")
            if param == "optimizer":
                try:
                    exec(f"from torch.optim import {value}")
                except Exception as e:
                    raise e
                else:
                    exec(f"values[{param}] = {value}")
            elif param == "nlinear":
                try:
                    exec(f"nn.{value}")
                except Exception as e:
                    raise e
                else:
                    exec(f"values[{param}] = nn.{value}")
            else:
                if not isinstance(value, type(defaults[i])):
                    raise TypeError(f"The type of argument '{param}' conficts with the expected type ({type(defaults[i])}).")
                values[param] = value
            defined.append(param)
        for param in set(tuple(params)).difference(set(tuple(defined))):
            values[param] = defaults[params.index(param)]

        return cls(**values)
