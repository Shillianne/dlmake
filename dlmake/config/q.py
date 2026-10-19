from typing import Literal, Optional
from dataclasses import dataclass


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


    @staticmethod
    def __check(obj: "QSettings"):
        if obj.source == "manual" and obj.value is None:
            raise ValueError(f"For Q configuration, a source of 'manual' can't have a Q value of None")
