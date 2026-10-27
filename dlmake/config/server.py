from dataclasses import dataclass
import warnings


params = {
    "host": (str, "127.0.0.1"),
    "port": (int, 38267),
    "vram_heartbeat": (float, 5.0),
}


@dataclass
class ServerSettings:
    use_server: bool
    host: str
    port: int
    vram_heartbeat: float

    @classmethod
    def from_dict(cls, **kwargs):
        keys = list(kwargs.keys())
        if (diff := set(keys).difference(set(params.keys()))):
            raise KeyError(f"Attribute '{list(diff)[0]}' not found in ServerSettings.")
        used = kwargs != {}
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
        values["use_server"] = used

        obj = cls(**values)
        cls.__check(obj, used)
        return obj

    @staticmethod
    def __check(obj: "ServerSettings", used: bool):
        if used:
            try:
                import fastapi
            except ImportError as e:
                warnings.warn(
                    message=(
                        "Server support requires FastAPI to be installed.\n"\
                        "Install the required libraries with: pip install 'dlmake[server]'."
                    ),
                    category=ImportWarning
                )
