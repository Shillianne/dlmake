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
loop_exit_probs: list[Queue[torch.Tensor]] = []
epoch: Queue[int] = Queue(maxsize=128)
eta: Queue[float] = Queue(maxsize=128)
