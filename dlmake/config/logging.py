from .benchmarks import BenchmarkSettings
from .metrics import MetricsSettings
from dataclasses import dataclass
from typing import Literal
import copy


params = {
    "verbose": (bool, True),
    "time_precision": (str, "ms"),
    "metrics": MetricsSettings,
    "benchmarks": BenchmarkSettings,
}

@dataclass
class LoggingSettings:
    verbose: bool
    time_precision: Literal["s", "ms"]
    metrics: MetricsSettings
    benchmarks: BenchmarkSettings

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
                values[param] = params[param].from_dict(kwargs[param])
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
            if isinstance(params[k], tuple):
                state[k] = v
            else:
                state[k] = v.state_dict()
        return state

    @classmethod
    def load_state_dict(cls, state: dict):
        d = copy.deepcopy(state)
        for k, v in state.items():
            if isinstance(v, dict):
                d[k] = params[k].load_state_dict(v)
        return cls(**d)
