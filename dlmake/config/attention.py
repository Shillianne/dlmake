from dataclasses import dataclass
from typing import Optional
import torch.nn as nn




@dataclass
class AttentionSettings:
    attention_method: nn.Module
    use_rope: bool
    use_qknorm: bool
    qk_norm_type: nn.Module
    n_kv_groups: Optional[int]
    q_head_dim: Optional[int]
    v_head_dim: Optional[int]
    kv_lora_dim: Optional[int]
    flash: bool
    # doc_aware: bool
    dropout: float
    bias: bool

    @classmethod
    def from_dict(cls, **kwargs):
        from ..transformer.attention import MultiHeadAttention, attention_types

        params = {
            "attention_method": (nn.Module, MultiHeadAttention),
            "use_rope": (bool, False),
            "use_qknorm": (bool, True),
            "qk_norm_type": (nn.Module, nn.RMSNorm),
            "n_kv_groups": ((None.__class__, int), None),
            "q_head_dim": ((None.__class__, int), None),
            "v_head_dim": ((None.__class__, int), None),
            "kv_lora_dim": ((None.__class__, int), None),
            "flash": (bool, False),
            # "doc_aware": (bool, False),
            "dropout": (float, 0.0),
            "bias": (bool, True),
        }

        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in AttentionSettings.")
        values = {}
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
                if param == "attention_method":
                    if value in (names := [att.__name__ for att in attention_types]):
                        exec(f"from ..transformer.attention import {value}")
                        exec(f"values['{param}'] = {value}")
                    elif value is None:
                        values[param] = MultiHeadAttention
                    else:
                        raise TypeError(f"The requested attention method ({value}) is not one of the available types ({names}).")
                elif param == "qk_norm_type":
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
