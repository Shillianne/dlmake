from dataclasses import dataclass
from typing import Literal


@dataclass
class LoopingSettings:
    n_max_loops: int
    looping_behaviour: Literal["full", "start", "middle", "end"]
