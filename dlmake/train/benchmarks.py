from dlmake import (
    Benchmark,
    HellaSwag,
    ARCEasy,
    ARCChallenge,
    PIQA,
    SciQ,
    Winogrande,
    OpenBookQA,
    CommonSenseQA,
    MMLU,
)
from dlmake import Transformer, Config
import torch.nn as nn
import tiktoken


class BenchmarkSuite:
    def __init__(
        self,
        config: Config, 
        savepath: str,
        device: str,
        loss_class: type[nn.Module],
    ):
        super(BenchmarkSuite, self).__init__()


        self.benchmarks: list[Benchmark] = []
        loss_options = config.training.loss_options.__dict__
        loss_options["reduction"] = "none"
        self.loss_fn = loss_class(**loss_options)
        if config.training.logging.benchmarks.hellaswag:
            self.benchmarks.append(HellaSwag(config, savepath, device, tiktoken.get_encoding("gpt2")))

        if config.training.logging.benchmarks.arc_easy:
            self.benchmarks.append(ARCEasy(config, savepath, device, tiktoken.get_encoding("gpt2")))

        if config.training.logging.benchmarks.arc_challenge:
            self.benchmarks.append(ARCChallenge(config, savepath, device, tiktoken.get_encoding("gpt2")))

        if config.training.logging.benchmarks.piqa:
            self.benchmarks.append(PIQA(config, savepath, device, tiktoken.get_encoding("gpt2")))

        if config.training.logging.benchmarks.sciq:
            self.benchmarks.append(SciQ(config, savepath, device, tiktoken.get_encoding("gpt2")))

        if config.training.logging.benchmarks.winogrande:
            self.benchmarks.append(Winogrande(config, savepath, device, tiktoken.get_encoding("gpt2")))

        if config.training.logging.benchmarks.openbookqa:
            self.benchmarks.append(OpenBookQA(config, savepath, device, tiktoken.get_encoding("gpt2")))

        if config.training.logging.benchmarks.commonsenseqa:
            self.benchmarks.append(CommonSenseQA(config, savepath, device, tiktoken.get_encoding("gpt2")))

        if config.training.logging.benchmarks.mmlu:
            self.benchmarks.append(MMLU(config, savepath, device, tiktoken.get_encoding("gpt2")))

    def prepare(self):
        for benchmark in self.benchmarks:
            benchmark.prepare()

    def evaluate(self, model: Transformer):
        results: dict[str, tuple[float, float]] = {}
        for benchmark in self.benchmarks:
            acc_norm, acc = benchmark(model, self.loss_fn)
            results[benchmark.name] = (acc, acc_norm)

        return results
