from dataclasses import dataclass


params = {
    "lr": (bool, False),
    "norm": (bool, False),
    "mfu": (bool, False),
    "time": (bool, False),
    "tok": (bool, False),
    "eta": (bool, False),
    "precision": (bool, False),
    "recall": (bool, False),
    "specificity": (bool, False),
    "f1score": (bool, False),
    "perplexity": (bool, False),
    "calibration_error": (bool, False),
    "topk_accuracy": (bool, False),
    "seqlen_loss": (bool, False),
}


@dataclass
class MetricsSettings:
    lr: bool
    norm: bool
    mfu: bool
    time: bool
    tok: bool
    eta: bool
    precision: bool
    recall: bool
    specificity: bool
    f1score: bool
    perplexity: bool
    calibration_error: bool
    topk_accuracy: bool
    seqlen_loss: bool

    @classmethod
    def from_dict(cls, *args):
        values = {}
        if args:
            for metric in args[0]:
                if metric not in params.keys():
                    raise KeyError(f"Attribute '{metric}' not found in MetricsSettings")
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
