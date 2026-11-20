from dataclasses import dataclass


params = {
    "hellaswag": (bool, False),
    "arc_easy": (bool, False),
    "arc_challenge": (bool, False),
    "piqa": (bool, False),
    "sciq": (bool, False),
    "winogrande": (bool, False),
    "openbookqa": (bool, False),
    "commonsenseqa": (bool, False),
    "mmlu": (bool, False),
}


@dataclass
class BenchmarkSettings:
    hellaswag: bool
    arc_easy: bool
    arc_challenge: bool
    piqa: bool
    sciq: bool
    winogrande: bool
    openbookqa: bool
    commonsenseqa: bool
    mmlu: bool

    @classmethod
    def from_dict(cls, *args):
        values = {}
        if args:
            for metric in args[0]:
                if metric not in params.keys():
                    raise KeyError(f"Attribute '{metric}' not found in BenchmarkSettings")
                values[metric] = True
        remaining = set(params.keys()).difference(set(values.keys()))
        for param in list(remaining):
            values[param] = False
        
        return cls(**values)
 
    def state_dict(self):
        return self.__dict__

    @classmethod
    def load_state_dict(cls, state: dict):
        return cls(**state)

