from torch.optim.lr_scheduler import LRScheduler, OneCycleLR
from typing import Optional, Type, Literal
from .general_params import GeneralParams
from torchmetrics import Accuracy, Metric
from torch.optim import Optimizer, AdamW
from .logging import LoggingSettings
from .server import ServerSettings
from dataclasses import dataclass
import torch.nn as nn


params = {
    "tokens_per_epoch": (int, 1024),
    "loaded_tokens": (int, 4096),
    "max_lr": (float, 1e-3),
    "min_lr": (float, 1e-6),
    "optimizer": ((Type[Optimizer], str), AdamW),
    "scheduler": ((Type[LRScheduler], str), OneCycleLR),
    "scheduler_options": GeneralParams,
    "loss": ((nn.Module, str), nn.CrossEntropyLoss),
    "accuracy": ((nn.Module, Metric, str), Accuracy),
    "use_fused": (bool, True),
    "compile": (bool, False),
    "checkpoint_rate": (int, 10),
    "save_history": (bool, False),
    "use_val": (bool, False),
    "train_size": (float, 0.8),
    "test_size": (float, 0.2),
    "val_size": (float, 0.0),
    "test_rate": ((None.__class__, int), None),
    "val_rate": ((None.__class__, int), None),
    "benchmark_rate": ((None.__class__, int), None),
    "train_epochs": (int, 12000),
    "warmup_steps": (int, 150),
    "shuffle_data": (bool, False),
    "new": (bool, True),
    "server": ServerSettings,
    "logging": LoggingSettings,
}


@dataclass
class TrainingSettings:
    tokens_per_epoch: int
    loaded_tokens: int
    max_lr: float
    min_lr: float
    optimizer: Type[Optimizer] | Literal["custom"]
    scheduler: Type[LRScheduler] | Literal["custom"]
    scheduler_options: GeneralParams
    loss: nn.Module | Literal["custom"]
    accuracy: nn.Module | Metric | Literal["custom"]
    use_fused: bool
    compile: bool
    checkpoint_rate: int
    save_history: bool
    use_val: bool
    train_size: float
    test_size: float
    val_size: float
    test_rate: Optional[int]
    val_rate: Optional[int]
    benchmark_rate: Optional[int]
    train_epochs: int
    warmup_steps: int
    shuffle_data: bool
    new: bool
    server: ServerSettings
    logging: LoggingSettings

    @classmethod
    def from_dict(cls, **kwargs):
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in TrainingSettings.")
        values = {}
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
                if param == "optimizer":
                    if value == "custom":
                        values[param] = value
                    else:
                        try:
                            exec(f"from torch.optim import {kwargs[param]}")
                        except Exception:
                            try:
                                exec(f"from bitsandbytes.optim import {kwargs[param]}")
                            except Exception as e:
                                raise e
                            else:
                                exec(f"values['{param}'] = {kwargs[param]}")
                        else:
                            exec(f"values['{param}'] = {kwargs[param]}")
                elif param == "scheduler":
                    if value == "custom":
                        values[param] = value
                    else:
                        try:
                            exec(f"from torch.optim.lr_scheduler import {kwargs[param]}")
                        except Exception as e:
                            raise e
                        else:
                            exec(f"values['{param}'] = {kwargs[param]}")
                elif param == "loss":
                    if param == "custom":
                        values[param] = value
                    else:
                        try:
                            exec(f"from torch.nn import {value}")
                        except Exception:
                            try:
                                exec(f"from dlmake import {value}")
                            except Exception as e:
                                raise e
                            else:
                                exec(f"values['{param}'] = {value}")
                        else:
                            exec(f"values['{param}'] = {value}")
                elif param == "accuracy":
                    if param == "custom":
                        values[param] = value
                    else:
                        try:
                            exec(f"from torchmetrics import {value}")
                        except Exception as e:
                            try:
                                exec(f"from dlmake import {value}")
                            except Exception as e:
                                raise e
                            else:
                                exec(f"values['{param}'] = {value}")
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
