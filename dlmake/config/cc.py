from dataclasses import dataclass
import copy


params = {
    "factor": (int, 2),
}

state_params = {
    "used": (bool, False),
    "factor": (int, 2),
}


@dataclass
class ContextCompressionSettings:
    used: bool
    factor: int

    @classmethod
    def from_dict(cls, **kwargs):
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in MoESettings.")
        values = {}
        used = kwargs != {}
        for param in keys:
            value = kwargs[param]
            # if isinstance(params[param], tuple):
            kind = params[param][0]
            # assert isinstance(kwargs[param], kind), f"Attribute '{param}' isn't of the expected type {kind}"
            assert issubclass(value.__class__, kind), f"Attribute '{param}' isn't of the expected type {kind}"
            values[param] = kwargs[param]
            # else:
            #     values[param] = params[param].from_dict(**kwargs[param])
        values["used"] = used
        remaining = set(params.keys()).difference(set(values.keys()))
        for param in list(remaining):
            # if isinstance(params[param], tuple):
            default = params[param][1]
            values[param] = default
            # else:
            #     values[param] = params[param].from_dict()

        return cls(**values)

    def state_dict(self):
        d = self.__dict__
        state = {}
        for k, v in d.items():
            if isinstance(state_params[k], tuple):
                state[k] = v
            else:
                state[k] = v.state_dict()
        return state

    @classmethod
    def load_state_dict(cls, state: dict):
        d = copy.deepcopy(state)
        for k, v in state.items():
            if isinstance(v, dict):
                d[k] = state_params[k].load_state_dict(v)
        return cls(**d)

