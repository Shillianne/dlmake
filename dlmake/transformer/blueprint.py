from abc import ABC, abstractmethod, ABCMeta
from dlmake import Config
import torch.nn as nn
import inspect
import torch

class TransformerMeta(ABCMeta, type):
    def __new__(cls, clsname: str, bases: tuple, attrs: dict):
        if nn.Module in bases:
            return type.__new__(cls, clsname, bases, attrs)
        if "__init__" not in attrs.keys():
            raise AttributeError("An __init__ method needs to be present within the '" + clsname + "' class")
        init = attrs["__init__"]
        def new_init(self, *args):
            init(self, *args)
            self.apply(self._init_weights)
        attrs["__init__"] = new_init
        return super(TransformerMeta, cls).__new__(cls, clsname, bases, attrs)
            

class Transformer(nn.Module, ABC, metaclass=TransformerMeta):
    def __init__(self, *args):
        super(Transformer, self).__init__()
        self.config = args[0]

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_SCALED_INIT"):
                std *= 2 * self.config.nlayers ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @abstractmethod
    def forward(self, *args) -> torch.Tensor:
        pass
    
    def configure_parameters(self, weight_decay: float, lr: float, device: str) -> torch.optim.Optimizer:
        param_dict = {name: param for name, param in self.named_parameters()}
        param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        no_decay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0}
        ]
        num_decayed_params = sum(p.numel() for p in decay_params)
        num_no_decayed_params = sum(p.numel() for p in no_decay_params)
        print(f"Number of decayed parameter tensors: {len(decay_params)}, with {num_decayed_params:,} parameters.")
        print(f"Number of non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decayed_params:,} parameters.")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = self.config.optimizer(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H, Q, T = cfg.nlayers, cfg.nhead, cfg.embed_dim // cfg.nhead, cfg.ctx_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_archived = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12 # A100 GPU bfloat16 peal flops is 312 TFLOPS
        mfu = flops_archived / flops_promised
        return mfu
