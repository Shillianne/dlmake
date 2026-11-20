from datasets import load_dataset, DatasetDict
from dlmake import Transformer, Config
from tiktoken import Encoding
from .base import Benchmark
import torch.nn as nn
import torch
import os


class CommonSenseQA(Benchmark):
    def __init__(
        self,
        config: Config,
        savepath: str,
        device: str,
        tokenizer: Encoding
    ):
        super(CommonSenseQA, self).__init__(
            config,
            savepath,
            device,
            tokenizer
        )
        self.savepath = os.path.join(os.path.dirname(self.savepath), ".cache", "benchmarks", "commonsenseqa")
        self.dataset = None
        self.name = "commonsenseqa"
        self.ctx = config.ctx_size
        self.__index = 0

    def prepare(self):
        if os.path.exists(self.savepath):
            self.dataset = DatasetDict.load_from_disk(self.savepath)["validation"]
        else:
            dataset = load_dataset("tau/commonsense_qa")
            if int(os.environ.get("DLMAKE_TESTING", -1)) == -1:
                dataset.save_to_disk(self.savepath)
            self.dataset = dataset["validation"]

    def __iter__(self):
        return self

    def __next__(self):
        assert self.dataset is not None, "Benchmark wasn't prepared or initialiazed"
        if self.__index >= len(self.dataset):  # we divide by 10 the whole benchmark so it runs in a reasonable amount of time
            self.__index = 0
            raise StopIteration()
        sample = self.dataset[self.__index]
        ctx: str = sample["question"]
        label: int = sample["choices"]["label"].index(sample["answerKey"])
        endings: list[str] = sample["choices"]["text"]

        ctx_tokens = self.tokenizer.encode(ctx)
        tok_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for ending in endings:
            end_tokens = self.tokenizer.encode(" " + ending.capitalize())
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

        max_len = max(len(end) for end in tok_rows)
        tokens = torch.zeros((5, max_len), dtype=torch.long)
        mask = torch.zeros((5, max_len), dtype=torch.long)
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
                losses = losses.view(5, -1)
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


