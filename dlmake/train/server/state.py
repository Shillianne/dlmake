from asyncio import Queue
import torch


loss: Queue[float] = Queue(maxsize=128)
acc: Queue[list[float]] = Queue(maxsize=128)
test_loss: Queue[float] = Queue(maxsize=128)
test_acc: Queue[list[float]] = Queue(maxsize=128)
val_loss: Queue[float] = Queue(maxsize=128)
val_acc: Queue[list[float]] = Queue(maxsize=128)
norm: Queue[float] = Queue(maxsize=128)
tkps: Queue[float] = Queue(maxsize=128)
mfu: Queue[float] = Queue(maxsize=128)
vram_usage: Queue[int] = Queue(maxsize=128)
gpu_usage: Queue[int] = Queue(maxsize=128)
loop_exit_probs: list[Queue[torch.Tensor]] = []
epoch: Queue[int] = Queue(maxsize=128)
eta: Queue[float] = Queue(maxsize=128)
lr: Queue[float] = Queue(maxsize=128)
train_time: Queue[float] = Queue(maxsize=128)
test_time: Queue[float] = Queue(maxsize=128)
val_time: Queue[float] = Queue(maxsize=128)
benchmark_time: Queue[float] = Queue(maxsize=128)
train_precision: Queue[float] = Queue(maxsize=128)
train_recall: Queue[float] = Queue(maxsize=128)
train_specificity: Queue[float] = Queue(maxsize=128)
train_f1score: Queue[float] = Queue(maxsize=128)
train_perplexity: Queue[float] = Queue(maxsize=128)
train_calibration_error: Queue[float] = Queue(maxsize=128)

test_precision: Queue[float] = Queue(maxsize=128)
test_recall: Queue[float] = Queue(maxsize=128)
test_specificity: Queue[float] = Queue(maxsize=128)
test_f1score: Queue[float] = Queue(maxsize=128)
test_perplexity: Queue[float] = Queue(maxsize=128)
test_calibration_error: Queue[float] = Queue(maxsize=128)

val_precision: Queue[float] = Queue(maxsize=128)
val_recall: Queue[float] = Queue(maxsize=128)
val_specificity: Queue[float] = Queue(maxsize=128)
val_f1score: Queue[float] = Queue(maxsize=128)
val_perplexity: Queue[float] = Queue(maxsize=128)
val_calibration_error: Queue[float] = Queue(maxsize=128)

test_top5_acc: Queue[float] = Queue(maxsize=128)
test_top10_acc: Queue[float] = Queue(maxsize=128)
test_top50_acc: Queue[float] = Queue(maxsize=128)
val_top5_acc: Queue[float] = Queue(maxsize=128)
val_top10_acc: Queue[float] = Queue(maxsize=128)
val_top50_acc: Queue[float] = Queue(maxsize=128)

test_seqlen_loss: list[Queue[float]] = [Queue(maxsize=128) for _ in range(8)]
val_seqlen_loss: list[Queue[float]] = [Queue(maxsize=128) for _ in range(8)]


hellaswag: Queue[tuple[float, float]] = Queue(maxsize=128)
arc_easy: Queue[tuple[float, float]] = Queue(maxsize=128)
arc_challenge: Queue[tuple[float, float]] = Queue(maxsize=128)
piqa: Queue[tuple[float, float]] = Queue(maxsize=128)
sciq: Queue[tuple[float, float]] = Queue(maxsize=128)
winogrande: Queue[tuple[float, float]] = Queue(maxsize=128)
openbookqa: Queue[tuple[float, float]] = Queue(maxsize=128)
commonsenseqa: Queue[tuple[float, float]] = Queue(maxsize=128)
mmlu: Queue[tuple[float, float]] = Queue(maxsize=128)
