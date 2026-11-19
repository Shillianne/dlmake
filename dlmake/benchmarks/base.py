from dlmake import Transformer, Config
from abc import ABC, abstractmethod
import torch.nn as nn
import tiktoken
import torch

class Benchmark(ABC):
    """
    Class to contain the information about a benchmark.
    Arguments:
        path: str (The path to the directory containing the desired benchmark)
    """
    def __init__(
        self,
        config: Config,
        path: str,
        device: str,
        tokenizer: tiktoken.Encoding
    ):
        self.config = config
        self.savepath = path
        self.device = device
        self.tokenizer = tokenizer
        self.name = "benchmark"

    @abstractmethod
    def prepare(self):
        ...

    @abstractmethod
    def evaluate(self, model: Transformer, loss_fn: nn.Module) -> tuple[float, float]:
        ...

    @torch.no_grad()
    def __call__(self, model: Transformer, loss_fn: nn.Module):
        return self.evaluate(model, loss_fn)
