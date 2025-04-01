from .transformer.tokenization import Tokenizer
from tiktoken import Encoding
from typing import Union
import torch
import json
import os

class Benchmark:
    """
    Class to contain the information about a benchmark.
    Arguments:
        path: str (The path to the directory containing the desired benchmark)
    """
    def __init__(self, name: str, path: str, tokenizer: Union[Tokenizer, Encoding]):
        self.name = name
        self.path = path
        self.tokenizer = tokenizer

    def iterate(self, split: str = 'val'):
        assert split == 'val', "Cannot perform benchmarks on splits other than validation for now."
        with open(os.path.join(self.path, f'{self.name}_{split}.jsonl'), 'r') as f:
            for line in f:
                example = json.loads(line)
                ctx = example["ctx"]  # Context for the test
                label = example["label"]  # The correct continuation out of the options
                endings = example["endings"]  # A list of endings for each of the options

                data = {
                    "label": label,
                    "ctx_tokens": None,
                    "ending_tokens": [],
                }

                ctx_tokens = self.tokenizer.encode(ctx)
                data['ctx_tokens'] = ctx_tokens
                tok_rows = []
                mask_rows = []
                for end in endings:
                    end_tokens = self.tokenizer.encode(" " + end)
                    tok_rows.append(ctx_tokens + end_tokens)
                    mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
                    data['ending_tokens'].append(end_tokens)

                max_len = max(len(row) for row in tok_rows)
                tokens = torch.zeros((4, max_len), dtype=torch.long)
                mask = torch.zeros((4, max_len), dtype=torch.long)
                for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
                    tokens[i, :len(tok_row)] = torch.tensor(tok_row)
                    mask[i, :len(mask_row)] = torch.tensor(mask_row)

                yield data, tokens, mask, label
