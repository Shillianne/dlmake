from dlmake import Transformer, Config
from tiktoken import Encoding
from .base import Benchmark
import torch.nn as nn
from tqdm import tqdm
import requests
import tempfile
import torch
import json
import os


class PIQA(Benchmark):
    def __init__(
        self,
        config: Config,
        savepath: str,
        device: str,
        tokenizer: Encoding
    ):
        super(PIQA, self).__init__(
            config,
            savepath,
            device,
            tokenizer
        )
        self.savepath = os.path.join(os.path.dirname(self.savepath), ".cache", "benchmarks", "piqa")
        self.dataset = None
        self.name = "piqa"
        self.ctx = config.ctx_size
        self.__index = 0
        self.__full_idx = 0

    @staticmethod
    def __load_dataset(dirname: str):
        with open(os.path.join(dirname, "valid-labels.lst"), "r") as f:
            with open(os.path.join(dirname, "valid.jsonl"), "r") as jf:
                dataset = []
                for label, line in zip(f, jf):
                    data = json.loads(line)
                    data["label"] = int(label)
                    dataset.append(data)
        return dataset

    def prepare(self):
        if os.path.exists(self.savepath):
            self.dataset = self.__load_dataset(self.savepath)
        else:
            files: list[str] = [
                "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/tests.jsonl",
                "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train-labels.lst",
                "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/train.jsonl",
                "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid-labels.lst",
                "https://raw.githubusercontent.com/ybisk/ybisk.github.io/master/piqa/data/valid.jsonl",
            ]
            if int(os.environ.get("DLMAKE_TESTING", -1)) != -1:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for file in files[-2:]:
                        response = requests.get(file, stream=True)
                        response.raise_for_status()
                        total = int(response.headers.get("Content-Length", 0))
                        with open(os.path.join(tmp_dir, os.path.basename(file)), "wb") as f, tqdm(
                            desc=os.path.basename(file),
                            total=total,
                            unit="1B",
                            unit_scale=True,
                            unit_divisor=1024,
                        ) as bar:
                            for data in response.iter_content(chunk_size=1024):
                                size = f.write(data)
                                bar.update(size)
                    self.dataset = self.__load_dataset(tmp_dir)
            else:
                for file in files:
                    response = requests.get(file, stream=True)
                    response.raise_for_status()
                    total = int(response.headers.get("Content-Length", 0))
                    with open(os.path.join(self.savepath, os.path.basename(file)), "wb") as f, tqdm(
                        desc=os.path.basename(file),
                        total=total,
                        unit="1B",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for data in response.iter_content(chunk_size=1024):
                            size = f.write(data)
                            bar.update(size)
                self.dataset = self.__load_dataset(self.savepath)

    def __iter__(self):
        return self

    def __next__(self):
        assert self.dataset is not None, "Benchmark wasn't prepared or initialiazed"
        if self.__index >= int(len(self.dataset) / 2) * (self.__full_idx + 1):  # we divide by 10 the whole benchmark so it runs in a reasonable amount of time
            self.__full_idx += 1
            if self.__full_idx == 2:
                self.__index = 0
                self.__full_idx = 0
            raise StopIteration()
        sample = self.dataset[self.__index]
        ctx: str = sample["goal"]
        label: int = sample["label"]
        endings: list[str] = [sample["sol1"], sample["sol2"]]

        ctx_tokens = self.tokenizer.encode(ctx)
        tok_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for ending in endings:
            end_tokens = self.tokenizer.encode(" " + ending)
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

        max_len = max(len(end) for end in tok_rows)
        tokens = torch.zeros((2, max_len), dtype=torch.long)
        mask = torch.zeros((2, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)
        self.__index += 1
        return tokens, mask, label

    def evaluate(self, model: Transformer, loss_fn: nn.Module):
        num_correct = 0
        num_correct_norm = 0
        num_total = 0
        for i, (tokens, mask, label) in enumerate(self):
            tokens = tokens.to(self.device)[:, :self.ctx]
            mask = mask.to(self.device)[:, :self.ctx]

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                logits = model(tokens)
                logits = (logits[..., :-1, :]).contiguous()
                tokens = (tokens[..., 1:]).contiguous()
                if not self.config.use_looping:
                    losses = loss_fn(logits.view(-1, logits.size(-1)), tokens.view(-1))
                else:
                    losses = loss_fn(logits, tokens, model.unnormalized[..., 1:, :])  # type: ignore
                losses = losses.view(2, -1)
            # now get the average loss just for the completion region (where mask == 1), in each row
            mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
            masked_losses = losses * mask
            # sum and divide by the number of 1s in the mask
            sum_loss = masked_losses.sum(dim=1)
            avg_loss = sum_loss / mask.sum(dim=1)
            # # now we have a loss for each of the 4 completions
            # # the one with the lowest loss should be the most likely
            pred = sum_loss.argmin().item()
            pred_norm = avg_loss.argmin().item()

            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)
            tokens = None
            mask = None
            torch.cuda.empty_cache()

        return num_correct_norm / num_total, num_correct / num_total

