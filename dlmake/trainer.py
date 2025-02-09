from torch.distributed import init_process_group, destroy_process_group
from typing import Callable, List, Optional, Type, Dict, Union, cast
from torch.nn.parallel import DistributedDataParallel as DDP
from dlmake import Config, BaseDataLoader
from dlmake.transformer import GPT
from torchmetrics import Accuracy
from .benchmark import Benchmark
import torch.distributed as dist
from .utils import get_most_likely_row, Logger
import torch.nn as nn
import inspect
import torch
import time
import os

Model = Union[GPT, nn.Module]

class Trainer:
    def __init__(self,
                 model: GPT,
                 dataloader: Type[BaseDataLoader],
                 config: Config,
                 savepath: str,
                 lr_schedule: Callable,
                 benchmarks: List[Benchmark],
                 step: int = 0,
                 tokens_per_batch: int = 0,
                 logger: Optional[Logger] = None)-> None:

        num_classes = config.vocab_size
        T = config.ctx_size

        self.ddp = int(os.environ.get("RANK", -1)) != -1
        self.config = config
        self.savepath = savepath
        self.benchmarks = benchmarks
        self.lr_scheduler: Callable = lr_schedule
        self.verbose = config.verbose
        self.batch_size = config.batch_size
        self.validation_rate = self.config.validation_rate if self.config.validation_rate is not None else 100
        self.benchmark_rate = (self.config.benchmark_rate if self.config.benchmark_rate is not None else 100) if self.benchmarks else 0
        self.logger = logger.log if logger is not None else (lambda x: print(x))
        self.loss_history: list[torch.Tensor] = []
        self.acc_history: list[torch.Tensor] = []
        self.norm_history: list[torch.Tensor] = []
        self.val_loss: list[torch.Tensor] = []
        self.val_acc: list[torch.Tensor] = []
        for benchmark in self.benchmarks:
            setattr(self, f"{benchmark}_acc", [])
        self.t = 1
        assert "reset" in dir(dataloader), "Provided dataloader needs to have a reset method"
        assert "split" in inspect.getfullargspec(dataloader).args, "The dataloader class needs to produce different loaders for training and validation."
        self.t = 1000 if config.time_precision == "ms" else 1
        self.compile = config.compile
        assert isinstance(self.compile, bool) and isinstance(self.config.new, bool), 'Compile and New have to be boolean arguments'

        if self.ddp:
            assert torch.cuda.is_available(), 'Cannot run DDP on CPU'
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.master_process = True
            self.device = 'cpu'
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            self.logger(f"Using device: {self.device}")

        self.train_loader = dataloader(batch_size=config.batch_size,
                                       context_size=T,
                                       epoch_batch_size=tokens_per_batch,
                                       split='train',
                                       rank=self.ddp_rank,
                                       world_size=self.ddp_world_size)
        if self.config.use_val:
            self.val_loader = dataloader(batch_size=config.batch_size,
                                         context_size=T,
                                         epoch_batch_size=tokens_per_batch,
                                         split='val',
                                         rank=self.ddp_rank,
                                         world_size=self.ddp_world_size)
        self.model = model
        if os.path.exists(os.path.join(self.savepath, 'snapshot.pth')) and not self.config.new:
            self._load_snapshot()
        else:
            self.model.to(self.device)
            if self.compile:
                self.logger("Starting model compilation...")
                self.model = cast(GPT, torch.compile(self.model))
                self.logger("Finished model compilation")
            if self.ddp:
                ddp_holder = DDP(self.model, device_ids=[self.ddp_local_rank])
                self.model = cast(GPT, ddp_holder.module)
            self.optimizer = self.model.configure_parameters(weight_decay=0.1, lr=6e4, device=self.device)
            self.step = step
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.accuracy.to(self.device)

    def train(self, epochs: int) -> None:
        assert epochs >= self.config.checkpoint_rate
        assert self.step < epochs, "Training was already completed for the requested number of epochs"
        running_mfu = -1.0

        for epoch in range(epochs):
            start = time.time()
            last_epoch = (epoch == epochs - 1)

            if (epoch % self.validation_rate == 0 or last_epoch) and self.config.use_val:
                self.model.eval()
                self.val_loader.reset()
                with torch.inference_mode():
                    val_loss = torch.tensor(0)
                    val_acc = torch.tensor(0)
                    for data in self.val_loader:
                        gpu_data = [v.to(self.device) for v in data]
                        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                            logits: torch.Tensor = self.model(*gpu_data[:-1])
                            loss: torch.Tensor = self.loss_fn(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                        loss = loss / self.val_loader.limit
                        acc: torch.Tensor = self.accuracy(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                        val_loss += loss.detach()
                        val_acc += acc.detach()
                    self.val_loss.append(val_loss)
                    self.val_acc.append(val_acc)
                    if self.ddp:
                        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                    if self.master_process:
                        self.logger(f"Validation loss: {val_loss.item():.3f} | Validation accuracy: {(val_acc.item() * 100):.3f}")

            if self.benchmarks and (epoch % self.benchmark_rate == 0 or last_epoch):
                self._validate_benchmarks()

            losses = torch.tensor(0)
            accs = torch.tensor(0)
            total = 0
            self.model.train()
            self.optimizer.zero_grad()

            for i, data in enumerate(self.train_loader):
                gpu_data = [v.to(self.device) for v in data]
                if self.ddp:
                    setattr(self.model, "require_backward_grad_sync", (i == self.train_loader.limit - 1))
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    y_preds: torch.Tensor = self.model(*gpu_data[:-1])
                    loss: torch.Tensor = self.loss_fn(y_preds.view(-1, y_preds.shape[-1]), gpu_data[-1].view(-1))
                loss = loss / self.train_loader.limit
                losses += loss.detach()
                acc: torch.Tensor = self.accuracy(y_preds.view(-1, y_preds.shape[-1]), gpu_data[-1].view(-1))
                accs += acc.detach()
                loss.backward()
                total = i
            if self.ddp:
                dist.all_reduce(losses, op=dist.ReduceOp.AVG)
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            lr = self.lr_scheduler(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.step()
            torch.cuda.synchronize()
            end = time.time()
            dt = end - start
            tok_per_sec = (self.train_loader.B * self.train_loader.T * self.train_loader.limit * self.ddp_world_size) / dt
            mfu = self.model.estimate_mfu(self.batch_size * self.train_loader.limit, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            accs /= (total + 1)
            if self.verbose and self.master_process:
                self.logger(f"[{epoch + 1}/{epoch}] | Average Loss: {losses:.3f} | Average Accuracy: {(accs * 100):.2f}% | lr: {lr:.3e} | Model Norm: {norm:.4f}"\
                      f" | MFU: {running_mfu * 100:.2f}% | dt: {(dt * self.t):.2f} {'s' if self.t == 1 else 'ms'} | tok/sec: {tok_per_sec:.2f}")
            self.loss_history.append(losses)
            self.acc_history.append(accs)
            self.norm_history.append(norm)
            
            if epoch % self.config.checkpoint_rate == 0 and self.master_process:
                self._save_snapshot(epoch)
        if self.master_process:
            self._save_snapshot(epochs - 1)
        if self.ddp:
            destroy_process_group()

    def _validate_benchmarks(self):
        num_correct_norm = 0
        num_total = 0
        results = {}
        for benchmark in self.benchmarks:
            for i, example in enumerate(benchmark.iterate()):
                if i % self.ddp_world_size != self.ddp_rank:
                    continue
                _, tokens, mask, label = example
                tokens = tokens.to(self.device)
                mask = mask.to(self.device)
                with torch.inference_mode():
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        logits: torch.Tensor = self.model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if self.ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=self.device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=self.device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            results[benchmark.name] = acc_norm
        s = []
        for benchmark, acc in results.items():
            s.append(f"{benchmark} Accuracy: {(acc * 100):.4f}%")
            benchmark_acc: list = getattr(self, f"{benchmark}_acc")
            benchmark_acc.append(acc)
            setattr(self, f"{benchmark}_acc", benchmark_acc)
        if self.master_process:
            self.logger(" | ".join(s))


    def _load_snapshot(self):
        snapshot = torch.load(os.path.join(self.savepath, 'snapshot.pth'))
        meta = torch.load(os.path.join(self.savepath, 'metadata.pth'))
        self.model.load_state_dict(snapshot)
        self.model.to(self.device)
        if self.compile:
            self.logger("Starting model compilation...")
            self.model = cast(GPT, torch.compile(self.model))
            self.logger("Finished model compilation")
        if self.ddp:
            ddp_holder = DDP(self.model, device_ids=[self.ddp_local_rank])
            self.model = cast(GPT, ddp_holder.module)

        self.step = meta["EPOCHS_RUN"]
        loader_settings: Dict[str, Dict] = meta["LOADER"]
        optim_state_dict = meta["OPTIM"]
        self.loss_history = meta["L_HISTORY"]
        self.acc_history = meta["A_HISTORY"]
        self.norm_history = meta["N_HISTORY"]
        self.val_loss = meta["VAL_LOSS"]
        self.val_acc = meta["VAL_ACC"]
        self.optimizer = self.model.configure_parameters(weight_decay=0.1, lr=6e-4, device=self.device)
        self.optimizer.load_state_dict(optim_state_dict)

        for benchmark in self.benchmarks:
            setattr(self, f'{benchmark}_acc', meta[f'{benchmark}_ACC'.upper()])

        self.train_loader.resume(*list(loader_settings['train'].values()))
        if self.config.use_val:
            self.val_loader.resume(*list(loader_settings['valid'].values()))
        self.logger("Loaded snapshot of the model.")
        
    def _save_snapshot(self, epoch: int):
        if not os.path.exists(self.savepath): os.mkdir(self.savepath)
        meta = {}
        meta['EPOCHS_RUN'] = epoch + 1
        if self.config.use_val:
            meta["LOADER"] = {
                "train": {
                    "iterations": self.train_loader.iterations,
                    "current_position": self.train_loader.current_position,
                    "current_shard": self.train_loader.current_shard
                },
                "valid": {
                    "iterations": self.val_loader.iterations,
                    "current_position": self.val_loader.current_position,
                    "current_shard": self.val_loader.current_shard
                }
            }
        else:
            meta["LOADER"] = {
                "train": {
                    "iterations": self.train_loader.iterations,
                    "current_position": self.train_loader.current_position,
                    "current_shard": self.train_loader.current_shard
                }
            }
        meta["OPTIM"] = self.optimizer.state_dict()
        meta["CONFIG"] = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        meta["L_HISTORY"] = self.loss_history
        meta["A_HISTORY"] = self.acc_history
        meta["N_HISTORY"] = self.norm_history
        meta["VAL_LOSS"] = self.val_loss
        meta["VAL_ACC"] = self.val_acc
        for benchmark in self.benchmarks:
            meta[f"{benchmark}_ACC".upper()] = getattr(self, f'{benchmark}_acc')
        torch.save(self.model.state_dict(), os.path.join(self.savepath, 'snapshot.pth'))
        torch.save(meta, os.path.join(self.savepath, 'metadata.pth'))
        self.logger("Saved snapshot of the model.")
