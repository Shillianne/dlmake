from typing import Literal, get_args
from dataclasses import dataclass
# from .kv import KVSettings
from .q import QSettings


params = {
    "n_max_loops": (int, 1),
    # "looping_behaviour": (Literal["full", "start", "middle", "end"], "full"),
    "looping_behaviour": (Literal["full", "end"], "full"),
    "q": QSettings,
    # "kv": KVSettings,
}


@dataclass
class LoopingSettings:
    n_max_loops: int
    # looping_behaviour: Literal["full", "start", "middle", "end"]
    looping_behaviour: Literal["full", "end"]
    q: QSettings
    # kv: KVSettings

    @classmethod
    def from_dict(cls, **kwargs):
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in LoopingSettings.")
        values = {}
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
                if param == "looping_behaviour":
                    if value not in get_args(params[param][0]):
                        raise ValueError(f"Attribute 'score_function' does not support a value of '{value}'.")
                    values[param] = value
                else:
                    kind = params[param][0]
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
