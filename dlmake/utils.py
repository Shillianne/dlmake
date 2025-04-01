import torch
from torch.nn import functional as F

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
    return torch.logical_and((seq_ids == seq_ids.T).transpose(0, 1), torch.tril(torch.ones((t.shape[1], t.shape[1]), dtype=t.dtype, device=t.device)))


__all__ = ["Logger", "get_most_likely_row"]
