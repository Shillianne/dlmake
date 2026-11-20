from dlmake import Config, BaseTransformer 
from dlmake import (
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
import torch.nn as nn
import tiktoken
import unittest
import torch
import os


class BenchmarkTest(unittest.TestCase):
    
    def test_hellaswag(self):
        os.environ["DLMAKE_TESTING"] = "1"
        config = Config.from_yaml("tests/configs/benchmark_model.yaml")
        model = BaseTransformer(config)
        tokenizer = tiktoken.get_encoding("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        benchmark = HellaSwag(config, "", device, tokenizer)
        benchmark.prepare()
        benchmark(model, loss_fn)

    def test_arc_easy(self):
        os.environ["DLMAKE_TESTING"] = "1"
        config = Config.from_yaml("tests/configs/benchmark_model.yaml")
        model = BaseTransformer(config)
        tokenizer = tiktoken.get_encoding("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        benchmark = ARCEasy(config, "", device, tokenizer)
        benchmark.prepare()
        benchmark(model, loss_fn)

    def test_arc_challenge(self):
        os.environ["DLMAKE_TESTING"] = "1"
        config = Config.from_yaml("tests/configs/benchmark_model.yaml")
        model = BaseTransformer(config)
        tokenizer = tiktoken.get_encoding("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        benchmark = ARCChallenge(config, "", device, tokenizer)
        benchmark.prepare()
        benchmark(model, loss_fn)

    def test_piqa(self):
        os.environ["DLMAKE_TESTING"] = "1"
        config = Config.from_yaml("tests/configs/benchmark_model.yaml")
        model = BaseTransformer(config)
        tokenizer = tiktoken.get_encoding("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        benchmark = PIQA(config, "", device, tokenizer)
        benchmark.prepare()
        benchmark(model, loss_fn)

    def test_sciq(self):
        os.environ["DLMAKE_TESTING"] = "1"
        config = Config.from_yaml("tests/configs/benchmark_model.yaml")
        model = BaseTransformer(config)
        tokenizer = tiktoken.get_encoding("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        benchmark = SciQ(config, "", device, tokenizer)
        benchmark.prepare()
        benchmark(model, loss_fn)

    def test_winogrande(self):
        os.environ["DLMAKE_TESTING"] = "1"
        config = Config.from_yaml("tests/configs/benchmark_model.yaml")
        model = BaseTransformer(config)
        tokenizer = tiktoken.get_encoding("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        benchmark = Winogrande(config, "", device, tokenizer)
        benchmark.prepare()
        benchmark(model, loss_fn)

    def test_openbookqa(self):
        os.environ["DLMAKE_TESTING"] = "1"
        config = Config.from_yaml("tests/configs/benchmark_model.yaml")
        model = BaseTransformer(config)
        tokenizer = tiktoken.get_encoding("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        benchmark = OpenBookQA(config, "", device, tokenizer)
        benchmark.prepare()
        benchmark(model, loss_fn)

    def test_commonsenseqa(self):
        os.environ["DLMAKE_TESTING"] = "1"
        config = Config.from_yaml("tests/configs/benchmark_model.yaml")
        model = BaseTransformer(config)
        tokenizer = tiktoken.get_encoding("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        benchmark = CommonSenseQA(config, "", device, tokenizer)
        benchmark.prepare()
        benchmark(model, loss_fn)

    def test_mmlu(self):
        os.environ["DLMAKE_TESTING"] = "1"
        config = Config.from_yaml("tests/configs/benchmark_model.yaml")
        model = BaseTransformer(config)
        tokenizer = tiktoken.get_encoding("gpt2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        benchmark = MMLU(config, "", device, tokenizer)
        benchmark.prepare()
        benchmark(model, loss_fn)
