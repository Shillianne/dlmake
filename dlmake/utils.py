import torch
from torch.nn import functional as F
import math

class Logger:
    def __init__(self, filename: str, new: bool = True):
        self.filename = filename
        if new:
            with open(filename, "w") as _:
                pass

    def log(self, msg: str):
        with open(self.filename, "a") as f:
            f.write(msg + "\n")
        print(msg)

def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

def document_aware_mask(t: torch.Tensor, eos: int) -> torch.Tensor:
    end_indeces = (t == eos).nonzero(as_tuple=True)
    seq_ids = torch.zeros_like(t, dtype=torch.long)
    eos_filter = torch.where(end_indeces[1] != t.shape[1] - 1)[0]
    end_indeces[1][eos_filter] += 1
    seq_ids[end_indeces[0][eos_filter], end_indeces[1][eos_filter]] = 1
    seq_ids = torch.unsqueeze(seq_ids.cumsum(dim=1), 0)
    pre_mask = torch.logical_and((seq_ids == seq_ids.permute(2, 1, 0)).transpose(0, 1), torch.tril(torch.ones((t.shape[1], t.shape[1]), dtype=t.dtype, device=t.device)))  # (batch, ctx, ctx)
    return torch.where(pre_mask == False, float("-inf"), 0)

def cosine_schedule(iteration: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int):
    if iteration < warmup_steps:
        return max_lr * (iteration + 1) / warmup_steps
    if iteration > max_steps:
        return min_lr
    decay_ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


__all__ = ["Logger", "get_most_likely_row"]
