from dataclasses import dataclass
from typing import Literal

progress_params = {
    "train_progress": (bool, False),
    "model_progress": (bool, False),
    "lr": (bool, False),
    "norm": (bool, False),
    "mfu": (bool, False),
    "time": (bool, False),
    "tok": (bool, False),
    "eta": (bool, False),
}


@dataclass
class ProgressConfig:
    train_progress: bool
    model_progress: bool
    lr: bool
    norm: bool
    mfu: bool
    time: bool
    tok: bool
    eta: bool

    @classmethod
    def from_dict(cls, **kwargs):
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(progress_params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in ProgressConfig.")
        values = {}
        for param in keys:
            kind = progress_params[param][0]
            assert isinstance(kwargs[param], kind), f"Attribute '{param}' isn't of the expected type {kind}"
            values[param] = kwargs[param]
        remaining = set(progress_params.keys()).difference(set(values.keys()))
        for param in list(remaining):
            default = progress_params[param][1]
            values[param] = default
        return cls(**values)


params = {
    "verbose": (bool, True),
    "time_precision": (str, "ms"),
    "val_progress_config": ProgressConfig,
    "test_progress_config": ProgressConfig,
    "train_progress_config": ProgressConfig,
}

@dataclass
class LoggingSettings:
    verbose: bool
    time_precision: Literal["s", "ms"]
    val_progress_config: ProgressConfig
    test_progress_config: ProgressConfig
    train_progress_config: ProgressConfig

    @classmethod
    def from_dict(cls, **kwargs):
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in LoggingSettings.")
        values = {}
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
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
