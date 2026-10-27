from dataclasses import dataclass
from typing import Literal


params = {
    "behavior": (Literal["accumulate", "last"], "accumulate"),
}


@dataclass
class KVSettings:
    behavior: Literal["accumulate", "last"]

    @classmethod
    def from_dict(cls, **kwargs):
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in KVSettings.")
        values = {}
        for param in keys:
            value = kwargs[param]
            kind = params[param][0]
            assert issubclass(value.__class__, kind), f"Attribute '{param}' isn't of the expected type {kind}"
            values[param] = kwargs[param]
        remaining = set(params.keys()).difference(set(values.keys()))
        for param in list(remaining):
            default = params[param][1]
            values[param] = default

        return cls(**values)
