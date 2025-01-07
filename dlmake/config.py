from dataclasses import is_dataclass
from typing import Type, Union, List, Dict, Any
import torch.nn as nn
from torch.optim import AdamW

Module = Type[nn.Module]

class ConfigMeta(type):
    def __call__(cls, *args, **kwargs):
        if cls.__base__ is not object and not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass.")
        return super().__call__(*args, **kwargs)

class Config(metaclass=ConfigMeta):
    ctx_size: int = 64
    vocab_size: int = 256
    embed_dim: int = 1
    nhead: int = 1
    nlayers: int = 1
    bias: bool = True
    flash: bool = False
    nlinear: Module = nn.ReLU
    nlinear_params: Union[List[Any], Dict[str, Any]]
    dropout_prob: float = 0.0
    optimizer = AdamW
