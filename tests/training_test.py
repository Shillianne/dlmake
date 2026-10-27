from dlmake import BaseDataLoader, Trainer, BaseTransformer, Config
from typing import Literal
import unittest
import torch
import os


class MockDataloader(BaseDataLoader):
    def __init__(
        self,
        config: Config,
        split: Literal["train", "test", "val"],
        min_idx: int,
        max_idx: int,
        rank: int,
        world_size: int,
    ):
        super().__init__(
            config,
            split,
            min_idx,
            max_idx,
            rank,
            world_size,
        )
        self.batch_accum = config.training.tokens_per_epoch // (config.batch_size * config.ctx_size * world_size)
        self.i = 0

    def reset(self):
        self.i = 0

    def resume(self, **state):
        raise NotImplementedError("Not used")

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.batch_accum:
            buf = torch.randint(0, 1024, (self.B, self.T + 1))
            X = buf[:, :-1]
            y = buf[:, 1:]
            self.i += 1
            return X, y
        else:
            self.i = 0
            raise StopIteration()


class TrainingTest(unittest.TestCase):
    
    def test_train_no_loop_no_server(self):
        os.environ["DLMAKE_TESTING"] = "1"
        trainer = Trainer(
            model=BaseTransformer,
            dataloader=MockDataloader,
            config_path="tests/configs/training_no_looping_no_server.yaml",
            datapath="",
            savepath="",
            benchmarks=[],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        trainer.train()

    def test_train_loop_no_server(self):
        os.environ["DLMAKE_TESTING"] = "1"
        trainer = Trainer(
            model=BaseTransformer,
            dataloader=MockDataloader,
            config_path="tests/configs/training_looping_no_server.yaml",
            datapath="",
            savepath="",
            benchmarks=[],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        trainer.train()

    def test_train_no_loop_server(self):
        os.environ["DLMAKE_TESTING"] = "1"
        trainer = Trainer(
            model=BaseTransformer,
            dataloader=MockDataloader,
            config_path="tests/configs/training_no_looping_server.yaml",
            datapath="",
            savepath="",
            benchmarks=[],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        trainer.train()

    def test_train_loop_server(self):
        os.environ["DLMAKE_TESTING"] = "1"
        trainer = Trainer(
            model=BaseTransformer,
            dataloader=MockDataloader,
            config_path="tests/configs/training_looping_server.yaml",
            datapath="",
            savepath="",
            benchmarks=[],
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        trainer.train()
