from typing import Literal, Optional
from dataclasses import dataclass
import copy


params = {
    "value": ((None.__class__, float), 0.9),
    "source": (Literal["manual", "data"], "manual"),
}


@dataclass
class QSettings:
    value: Optional[float]
    source: Literal["manual", "data"]

    @classmethod
    def from_dict(cls, **kwargs):
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in QSettings.")
        values = {}
        for param in keys:
            value = kwargs[param]
            if isinstance(params[param], tuple):
                kind = params[param][0]
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

        obj = cls(**values)
        cls.__check(obj)
        return obj

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

    @staticmethod
    def __check(obj: "QSettings"):
        if obj.source == "manual" and obj.value is None:
            raise ValueError(f"For Q configuration, a source of 'manual' can't have a Q value of None")
