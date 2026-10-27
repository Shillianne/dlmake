from dlmake import Transformer, Config, DecoderBlock, BaseDataLoader, Trainer
from typing import Literal
import torch.nn as nn
import torch
import os


class GPT2(Transformer):
    def __init__(self, config: Config, **kwargs):
        super(GPT2, self).__init__(config)

        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.ctx_size, config.embed_dim)

        self.blocks = nn.ModuleList([
            DecoderBlock(config, layer=i) for i in range(config.nlayers)
        ])

        self.norm = config.norm(config.embed_dim)
        self.classification_head = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)

    def forward(self, *args, start_pos: int = 0):
        (x,) = args
        b, t = x.shape

        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(t, device=x.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x, torch.arange(t, device=x.device))
        x = self.norm(x)
        return self.classification_head(x)



class ExampleDataloader(BaseDataLoader):
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


if __name__ == "__main__":
    os.environ["DLMAKE_TESTING"] = "1"  # Since the dataloader doesn't use any real dataset, we use a testing flag to avoid errors
    trainer = Trainer(
        model=GPT2,
        dataloader=ExampleDataloader,
        config_path="./config.yaml",
        datapath="",
        savepath="",
        benchmarks=[],
        device=("cuda" if torch.cuda.is_available() else "cpu")
    )
    trainer.train()
