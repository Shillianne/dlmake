from .base import Benchmark
from .hellaswag import HellaSwag
from .arc import ARCEasy, ARCChallenge
from .piqa import PIQA
from .sciq import SciQ
from .winogrande import Winogrande
from .openbookqa import OpenBookQA
from .commonsenseqa import CommonSenseQA
from .mmlu import MMLU


benchmarks: list[type[Benchmark]] = [
    HellaSwag,
    ARCEasy,
    ARCChallenge,
    PIQA,
    SciQ,
    Winogrande,
    OpenBookQA,
    CommonSenseQA,
    MMLU,
]

__all__ = [
    "Benchmark",
    "HellaSwag",
    "ARCEasy",
    "ARCChallenge",
    "PIQA",
    "SciQ",
    "Winogrande",
    "OpenBookQA",
    "CommonSenseQA",
    "MMLU",
]
