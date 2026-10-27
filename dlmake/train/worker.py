from dlmake import Transformer, Config, BaseDataLoader, SplitGenerator, Benchmark
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing.connection import Connection
from .utils import Logger, progress_factory
from typing import Type, cast, Optional
from torch.optim import Optimizer
import torch.distributed as dist
import torch.nn as nn
import torch
import time
import os


class Worker:
    def __init__(
        self,
        model_class: Type[Transformer],
        config: Config,
        datapath: str,
        savepath: str,
        dataloader: Type[BaseDataLoader],
        benchmarks: list[Benchmark],
        model_index: int,
        model_count: int,
        device: str,
        logger: Logger,
        **kwargs
    ):
        self.saves = 0
        self.model_class = model_class
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        self.testing = int(os.environ.get("DLMAKE_TESTING", -1)) != -1
        self.device = device
        self.logger = logger
        self.config = config
        self.datapath = datapath
        self.savepath = savepath
        self.dataloader = dataloader
        self.model_index = model_index
        self.model_count = model_count
        self.benchmarks = benchmarks
        self.custom_ff = None
        self.custom_shallow_ff = None
        self.optimizer_class = None
        self.scheduler_class = None
        self.loss_class = None
        self.accuracy_class = None
        self.loss_history: list[torch.Tensor] = []
        self.acc_history: list[torch.Tensor] = []
        self.norm_history: list[torch.Tensor] = []
        self.test_loss: list[torch.Tensor] = []
        self.test_acc: list[torch.Tensor] = []

        for k, v in kwargs.items():
            if k == "optimizer":
                if config.training.optimizer != "custom":
                    logger.warn("A custom optimizer has been provided but, one was already provided in the configuration. Ignoring custom optimizer.")
                    self.optimizer_class = config.training.optimizer
                else:
                    assert issubclass(v, torch.optim.Optimizer), "The custom provided optimizer has to be a subclass of 'Optimizer'."
                    self.optimizer_class = v
            elif k == "scheduler":
                if config.training.scheduler != "custom":
                    logger.warn("A custom learning rate scheduler has been provided, but one was already provided in the configuration. Ignoring custom scheduler.")
                    self.scheduler_class = config.training.scheduler
                else:
                    assert issubclass(v, torch.optim.lr_scheduler.LRScheduler), "The custom provided scheduler has to be a subclass of 'LRScheduler'."
                    self.scheduler_class = v
            elif k == "loss":
                if config.training.loss != "custom":
                    logger.warn("A custom loss function has been provided, but one was already provided in the configuration. Ignoring custom loss.")
                    self.loss_class = config.training.loss
                else:
                    assert issubclass(v, nn.Module), "The custom provided loss has to be a subclass of 'nn.Module'."
                    self.loss_class = v
            elif k == "accuracy":
                if config.training.accuracy != "custom":
                    logger.warn("A custom accuracy function has been provided, but one was already provided in the configuration. Ignoring custom accuracy.")
                    self.accuracy_class = config.training.accuracy
                else:
                    assert issubclass(v, nn.Module), "The custom provided accuracy has to be a subclass of 'nn.Module'."
                    self.accuracy_class = v
            elif k == "ff":
                if config.feed_forward.ff != "custom":
                    logger.warn("A custom Feed Forward network has been provided, but one was already provided in the configuration. Ignoring custom Feed Forward network.")
                else:
                    assert issubclass(v, nn.Module), "The cusom provided Feed Forward network has to be a subclass of 'nn.Module'."
                    self.custom_ff = v
            elif k == "shallow_ff":
                if config.feed_forward.shallow_ff != "custom":
                    logger.warn("A custom Shallow Feed Forward network has been provided, but one was already provided in the configuration. Ignoring custom Shallow Feed Forward network.")
                else:
                    assert issubclass(v, nn.Module), "The cusom provided Shallow Feed Forward network has to be a subclass of 'nn.Module'."
                    self.custom_shallow_ff = v

    def __init_model(self):
        try:
            model = self.model_class(self.config, ff=self.custom_ff, shallow_ff=self.custom_shallow_ff)
            model.to(self.device)
            if self.config.training.compile:
                self.logger.log("Starting model compilation...")
                model = cast(Transformer, torch.compile(model))
                self.logger.log("Finished model compilation.")
            if self.ddp:
                ddp_holder = DDP(model, device_ids=[self.ddp_local_rank])
                model = cast(Transformer, ddp_holder.module)
        except torch.cuda.OutOfMemoryError:
            self.logger.error("Couldn't allocate enough memory for the model")
            torch.cuda.empty_cache()
            return None
        else:
            return model

    def __train(self, pipe: Connection, model: Transformer):
        acc_time = 0
        running_mfu = 1.0
        t = 1000 if self.config.training.logging.time_precision == "ms" else 1
        train_size = self.config.training.train_size
        test_size = self.config.training.test_size
        train_print_f = progress_factory(self.config.training.logging.train_progress_config)
        test_print_f = progress_factory(self.config.training.logging.test_progress_config)
        test_rate = self.config.training.test_rate if self.config.training.test_rate is not None else 100
        val_rate = self.config.training.val_rate if self.config.training.val_rate is not None else 0
        if self.config.training.use_val:
            train_dataloader, test_dataloader, val_dataloader = SplitGenerator(
                data_path=self.datapath, use_val=True, train_size=train_size, test_size=test_size
            ).generate_splits(
                dataloader=self.dataloader,
                config=self.config,
                rank=self.ddp_rank,
                world_size=self.ddp_world_size,
            )
            val_print_f = progress_factory(self.config.training.logging.val_progress_config)
        else:
            val_dataloader = None
            val_print_f = None
            train_dataloader, test_dataloader = SplitGenerator(
                data_path=self.datapath, use_val=False, train_size=train_size, test_size=test_size
            ).generate_splits(
                dataloader=self.dataloader,
                config=self.config,
                rank=self.ddp_rank,
                world_size=self.ddp_world_size,
            )
        
        optimizer = model.configure_parameters(weight_decay=0.1, lr=6e-4, device=self.device, custom_optim=self.optimizer_class)
        if self.config.training.scheduler == "custom":
            assert self.scheduler_class is not None, "A custom scheduler wasn't provided even though the configuration claimed it so."
            scheduler = self.scheduler_class(optimizer)
        else:
            scheduler = self.config.training.scheduler(optimizer, **self.config.training.scheduler_options.__dict__)  # TODO: Add option for scheduler parameters
        if self.config.training.loss == "custom":
            assert self.loss_class is not None, "A custom loss wasn't provided even though the configuration claimed it so."
            loss_fn = self.loss_class()
        else:
            loss_fn = self.config.training.loss()
        if self.config.training.accuracy == "custom":
            assert self.accuracy_class is not None, "A custom accuracy wasn't provided even though the configuration claimed it so."
            if self.accuracy_class.__name__ == "Accuracy":
                accuracy = self.accuracy_class(task="multiclass", num_classes=self.config.vocab_size)
            else:
                accuracy = self.accuracy_class(self.config)
        else:
            if self.config.training.accuracy.__name__ == "Accuracy":
                accuracy = self.config.training.accuracy(task="multiclass", num_classes=self.config.vocab_size)
            else:
                accuracy = self.config.training.accuracy(self.config)
        accuracy.to(self.device)
        val_loss = None
        val_acc = None

        try:
            for epoch in range(self.config.training.train_epochs):
                start = time.time()
                last_epoch = (epoch == self.config.training.train_epochs - 1)

                if (epoch % test_rate == 0 or last_epoch):
                    # model.eval()
                    with torch.inference_mode():
                        test_loss = torch.tensor(0., device=self.device)
                        if self.config.use_looping:
                            test_acc = torch.zeros(self.config.looping.n_max_loops, device=self.device)
                        else:
                            test_acc = torch.tensor(0., device=self.device)
                        i = 0
                        for i, data in enumerate(test_dataloader):
                            gpu_data = [v.to(self.device) if i != len(data) - 1 else v.to(self.device).contiguous() for i, v in enumerate(data)]
                            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                                logits: torch.Tensor = model(*gpu_data[:-1])
                                if self.config.use_looping:
                                    loss: torch.Tensor = loss_fn(
                                        logits, 
                                        gpu_data[-1],
                                        model.unnormalized,
                                    )
                                else:
                                    loss: torch.Tensor = loss_fn(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                            loss /= self.limit
                            if self.config.use_looping:
                                acc: torch.Tensor = accuracy(logits, gpu_data[-1])
                            else:
                                acc: torch.Tensor = accuracy(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                            test_loss += loss.detach()
                            test_acc += acc.detach()
                        test_acc /= (i + 1)
                        if self.ddp:
                            dist.all_reduce(test_loss)
                        if self.master_process:
                            self.logger.log(test_print_f(loss=test_loss.item(), acc=test_acc, looping=self.config.use_looping))

                    self.test_loss.append(test_loss)
                    self.test_acc.append(test_acc)

                    if self.master_process:
                        pipe.send({
                            "test_loss": test_loss.item(),
                            "test_acc": [test_acc.item()] if not self.config.use_looping else test_acc.tolist(),
                        })

                if self.config.training.use_val and (epoch % val_rate == 0 or last_epoch):
                    assert val_dataloader is not None
                    assert val_print_f is not None
                    # model.eval()
                    with torch.inference_mode():
                        val_loss = torch.tensor(0., device=self.device)
                        if self.config.use_looping:
                            val_acc = torch.zeros(self.config.looping.n_max_loops, device=self.device)
                        else:
                            val_acc = torch.tensor(0., device=self.device)
                        i = 0
                        for i, data in enumerate(val_dataloader):
                            gpu_data = [v.to(self.device) if i != len(data) - 1 else v.to(self.device).contiguous() for i, v in enumerate(data)]
                            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                                logits: torch.Tensor = model(*gpu_data[:-1])
                                if self.config.use_looping:
                                    loss: torch.Tensor = loss_fn(
                                        logits, 
                                        gpu_data[-1],
                                        model.unnormalized,
                                    )
                                else:
                                    loss: torch.Tensor = loss_fn(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                            loss /= self.limit
                            if self.config.use_looping:
                                acc: torch.Tensor = accuracy(logits, gpu_data[-1])
                            else:
                                acc: torch.Tensor = accuracy(logits.view(-1, logits.shape[-1]), gpu_data[-1].view(-1))
                            val_loss += loss.detach()
                            val_acc += acc.detach()
                        val_acc /= (i + 1)
                        if self.ddp:
                            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                        if self.master_process:
                            self.logger.log(val_print_f(loss=val_loss.item(), acc=val_acc, looping=self.config.use_looping))

                    if self.master_process:
                        pipe.send({
                            "val_loss": val_loss.item(),
                            "val_acc": [val_acc.item()] if not self.config.use_looping else val_acc.tolist(),
                        })
                else:
                    val_loss = None
                    val_acc = None

                losses = torch.tensor(0., device=self.device)
                if self.config.use_looping:
                    accs = torch.zeros(self.config.looping.n_max_loops, device=self.device)
                    unnormalized = torch.zeros(self.config.looping.n_max_loops, self.config.batch_size, self.config.ctx_size, 1, device=self.device)
                else:
                    unnormalized = torch.tensor(0.)
                    accs = torch.tensor(0., device=self.device)
                total = 0
                model.train()
                optimizer.zero_grad()

                for i, data in enumerate(train_dataloader):
                    gpu_data = [v.to(self.device) if i != len(data) - 1 else v.to(self.device).contiguous() for i, v in enumerate(data)]
                    if self.ddp:
                        setattr(model.inner, "require_backward_grad_sync", (i == self.limit - 1))
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        y_preds: torch.Tensor = model(*gpu_data[:-1])
                        if self.config.use_looping:
                            loss: torch.Tensor = loss_fn(
                                y_preds,
                                gpu_data[-1],
                                model.unnormalized,
                            )
                        else:
                            loss: torch.Tensor = loss_fn(y_preds.view(-1, y_preds.shape[-1]), gpu_data[-1].view(-1))
                    if self.config.use_looping:
                        unnormalized += model.unnormalized.detach()  # type: ignore
                    loss /= self.limit
                    losses += loss.detach()
                    if self.config.use_looping:
                        acc: torch.Tensor = accuracy(y_preds, gpu_data[-1])
                    else:
                        acc: torch.Tensor = accuracy(y_preds.view(-1, y_preds.shape[-1]), gpu_data[-1].view(-1))
                    accs += acc.detach()
                    loss.backward()
                    total = i
                if self.ddp:
                    dist.all_reduce(losses, op=dist.ReduceOp.AVG)
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                end = time.time()
                dt = end - start
                tok_per_sec = (train_dataloader.B * train_dataloader.T * self.limit * self.ddp_world_size) / dt
                mfu = model.estimate_mfu(self.config.batch_size * self.limit, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                acc_time += dt
                seconds = ((self.config.training.train_epochs * acc_time) / (epoch + 1)) - acc_time  # expectation - evidence
                lr = scheduler.get_last_lr()[0]
                self.loss_history.append(losses)
                self.acc_history.append(accs)
                self.norm_history.append(norm)
                
                accs /= (total + 1)
                if self.config.use_looping:
                    unnormalized /= (total + 1)
                if self.config.training.logging.verbose and self.master_process:
                    self.logger.log(train_print_f(
                        loss=losses, 
                        acc=accs,
                        epoch=epoch + 1,
                        total_epochs=self.config.training.train_epochs,
                        cur_model=self.model_index,
                        total_models=self.model_count,
                        lr=lr,
                        norm=norm,
                        mfu=running_mfu,
                        time=dt,
                        scale=t,
                        tokens=tok_per_sec,
                        eta=seconds,
                        looping=self.config.use_looping,
                    ))

                if self.master_process:
                    pipe.send({
                        "loss": losses.item(),
                        "acc": [accs.item()] if not self.config.use_looping else accs.tolist(),
                        "norm": norm.item(),
                        "tokps": tok_per_sec,
                        "mfu": running_mfu,
                        "loop_exit_probs": (unnormalized if self.config.use_looping else None),
                        "epoch": epoch,
                        "eta": seconds,
                    })
                
                if epoch % self.config.training.checkpoint_rate == 0 and self.master_process and not self.testing:
                    self.__save(
                        model,
                        epoch,
                        train_dataloader,
                        test_dataloader,
                        val_dataloader,
                        optimizer,
                        val_loss,
                        val_acc
                    )

            if self.master_process and not self.testing:
                self.__save(
                    model,
                    self.config.training.train_epochs,
                    train_dataloader,
                    test_dataloader,
                    val_dataloader,
                    optimizer,
                    val_loss,
                    val_acc
                )

        except torch.cuda.OutOfMemoryError:
            self.logger.error(
                f"Ran out of CUDA memory on model [{self.model_index + 1:>{len(str(self.model_count))}}/{self.model_count}]."
            )
        else:
            self.logger.log(
                f"Finished training model [{self.model_index + 1:>{len(str(self.model_count))}}/{self.model_count}]."
            )
        finally:
            if self.ddp:
                destroy_process_group()
            pipe.send("end")


    def work(self, pipe: Connection):
        if self.ddp:
            assert torch.cuda.is_available(), "Cannot run DDP on CPU"
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0
        else:
            self.ddp_rank = 0
            self.ddp_local_rank = 0
            self.ddp_world_size = 1
            self.master_process = True
            if self.device == "cuda" and torch.cuda.is_available():
                self.device = "cuda"
            elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        self.logger.log(f"Running on device: {self.device}")
        
        model = self.__init_model()
        if model is None:
            pipe.send("end")
            return

        assert isinstance(model.blocks, nn.ModuleList)
        self.logger.log(
            f"Using attention method: {model.blocks[0].attn.__class__.__name__}"
        )
        self.limit = self.config.training.tokens_per_epoch // (
            self.config.batch_size * self.config.ctx_size * self.ddp_world_size
        )

        self.__train(pipe, model)

    def __save(
        self,
        model: Transformer,
        epoch: int,
        train_dataloader: BaseDataLoader,
        test_dataloader: BaseDataLoader,
        val_dataloader: Optional[BaseDataLoader],
        optimizer: Optimizer,
        val_loss: Optional[torch.Tensor],
        val_acc: Optional[torch.Tensor],
    ):
        meta = {}
        meta['EPOCHS_RUN'] = epoch + 1
        if self.config.training.use_val:
            meta["LOADER"] = {
                "train": train_dataloader.__dict__,
                "test": test_dataloader.__dict__,
                "val": val_dataloader.__dict__,
            }
            meta["VAL_LOSS"] = val_loss
            meta["VAL_ACC"] = val_acc
        else:
            meta["LOADER"] = {
                "train": train_dataloader.__dict__,
                "test": test_dataloader.__dict__,
            }
        meta["OPTIM"] = optimizer.state_dict()
        meta["CONFIG"] = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        meta["L_HISTORY"] = self.loss_history
        meta["A_HISTORY"] = self.acc_history
        meta["N_HISTORY"] = self.norm_history
        meta["TEST_LOSS"] = self.test_loss
        meta["TEST_ACC"] = self.test_acc
        for benchmark in self.benchmarks:
            meta[f"{benchmark}_ACC".upper()] = getattr(self, f'{benchmark}_acc')
        if not self.config.training.save_history:
            torch.save(model.state_dict(), os.path.join(self.savepath, 'snapshot.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(self.savepath, "history", f"snapshot_{self.saves:>03}.pth"))
            self.saves += 1
        torch.save(meta, os.path.join(self.savepath, 'metadata.pth'))
        self.logger.log("Saved snapshot of the model.")
