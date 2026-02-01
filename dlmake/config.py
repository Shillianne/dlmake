from typing import Literal, Type, Union, List, Dict, Any, Optional
from torch.optim import AdamW, Optimizer
import torch.nn as nn
import json

Module = Type[nn.Module]

progress_base = {
    "train_progress": False,
    "model_progress": False,
    "lr": False,
    "norm": False,
    "mfu": False,
    "time": False,
    "tok": False,
    "eta": False
}

class Config:
    def __init__(self,
                 ctx_size: int = 64,
                 max_ctx_size: int = 256,
                 vocab_size: int = 256,
                 embed_dim: int = 1,
                 batch_size: int = 1,
                 max_batch_size: int = 2,
                 tokens_per_batch: int = 1024,
                 nhead: int = 1,
                 nlayers: int = 1,
                 bias: bool = True,
                 norm: Module = nn.LayerNorm,
                 norm_params: dict[str, Union[int, float]] = {},
                 nlinear: Module = nn.ReLU,
                 nlinear_params: Union[List[Any], Dict[str, Any]] = [],
                 dropout_prob: float = 0.0,
                 attention_method: Optional[type] = None,
                 n_kv_groups: Optional[int] = None,
                 q_head_dim: Optional[int] = None,
                 kv_lora_dim: Optional[int] = None,
                 v_head_dim: Optional[int] = None,
                 max_lr: float = 1e-3,
                 min_lr: float = 1e-6,
                 optimizer = AdamW,
                 flash: bool = False,
                 doc_aware: bool = False,
                 verbose: bool = True,
                 compile: bool = False,
                 checkpoint_rate: int = 10,
                 use_val: bool = False,
                 train_size: float = 0.8,
                 test_size: float = 0.2,
                 val_size: float = 0.0,
                 test_rate: Optional[int] = None,
                 validation_rate: Optional[int] = None,
                 benchmark_rate: Optional[int] = None,
                 train_epochs: int = 12000,
                 warmup_steps: int = 150,
                 shuffle_data: bool = False,
                 new: bool = True,
                 time_precision: Literal['s', 'ms'] = 'ms',
                 val_progress_config: dict[str, bool] = progress_base,
                 test_progress_config: dict[str, bool] = progress_base,
                 train_progress_config: dict[str, bool] = progress_base):
        self.ctx_size = ctx_size
        self.max_ctx_size = max_ctx_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.tokens_per_batch = tokens_per_batch
        self.nhead = nhead
        self.nlayers = nlayers
        self.bias = bias
        self.norm = norm
        self.norm_params = norm_params
        self.nlinear = nlinear
        self.nlinear_params = nlinear_params
        self.dropout_prob = dropout_prob
        self.attention_method = attention_method
        self.n_kv_groups = n_kv_groups
        self.q_head_dim = q_head_dim
        self.kv_lora_dim= kv_lora_dim
        self.v_head_dim = v_head_dim
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.optimizer = optimizer
        self.flash = flash
        self.doc_aware: bool = doc_aware
        self.verbose = verbose
        self.compile = compile
        self.checkpoint_rate = checkpoint_rate
        self.use_val = use_val
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.test_rate = test_rate
        self.validation_rate = validation_rate
        self.benchmark_rate = benchmark_rate
        self.train_epochs = train_epochs
        self.warmup_steps = warmup_steps
        self.shuffle_data = shuffle_data
        self.new = new
        self.time_precision = time_precision
        self.val_progress_config = val_progress_config
        self.test_progress_config = test_progress_config
        self.train_progress_config = train_progress_config

    @classmethod
    def from_json(cls, filename: str) -> "Config":
        from .transformer.attention import MultiHeadAttention, attention_types
        with open(filename, "r") as f:
            config = json.load(f)
        params = ["ctx_size", "max_ctx_size", "vocab_size", "embed_dim", "batch_size", "max_batch_size", "tokens_per_batch", "nhead", "nlayers",
                  "bias", "norm", "norm_params", "nlinear", "nlinear_params", "attention_method", "n_kv_groups", "q_head_dim", "kv_lora_dim", "v_head_dim",
                  "dropout_prob", "max_lr", "min_lr", "optimizer", "flash", "doc_aware", "verbose", "compile", "checkpoint_rate", "use_val", "train_size", 
                  "test_size", "val_size", "test_rate", "validation_rate", "benchmark_rate", "train_epochs", "warmup_steps", "shuffle_data", "new",
                  "time_precision", "val_progress_config", "test_progress_config", "train_progress_config"]
        default_types = {
            "ctx_size": int,
            "max_ctx_size": int,
            "vocab_size": int,
            "embed_dim": int, 
            "batch_size": int,
            "max_batch_size": int,
            "tokens_per_batch": int,
            "nhead": int,
            "nlayers": int,
            "bias": bool,
            "norm": nn.Module,
            "norm_params": dict,
            "nlinear": nn.Module,
            "nlinear_params": dict,
            "dropout_prob": float,
            "attention_method": (None.__class__, type),
            "n_kv_groups": (None.__class__, int),
            "q_head_dim": (None.__class__, int),
            "kv_lora_dim": (None.__class__, int),
            "v_head_dim": (None.__class__, int),
            "max_lr": float,
            "min_lr": float,
            "optimizer": Optimizer,
            "flash": bool,
            "doc_aware": bool,
            "verbose": bool,
            "compile": bool,
            "checkpoint_rate": int,
            "use_val": bool,
            "train_size": float,
            "test_size": float,
            "val_size": float,
            "test_rate": (None.__class__, int),
            "validation_rate": (None.__class__, int),
            "benchmark_rate": (None.__class__, int),
            "train_epochs": int,
            "warmup_steps": int,
            "shuffle_data": bool,
            "new": bool,
            "time_precision": str,
            "val_progress_config": dict,
            "test_progress_config": dict,
            "train_progress_config": dict
        }
        defaults = {
            "ctx_size": 64,
            "max_ctx_size": 256,
            "vocab_size": 256,
            "embed_dim": 1,
            "batch_size": 1,
            "max_batch_size": 2,
            "tokens_per_batch": 1024,
            "nhead": 1,
            "nlayers": 1,
            "bias": True,
            "norm": nn.LayerNorm,
            "norm_params": {},
            "nlinear": nn.ReLU,
            "nlinear_params": {},
            "dropout_prob": 0.0,
            "attention_method": None,
            "n_kv_groups": None,
            "q_head_dim": None,
            "kv_lora_dim": None,
            "v_head_dim": None,
            "max_lr": 1e-3,
            "min_lr": 1e-6,
            "optimizer": AdamW,
            "flash": False,
            "doc_aware": False,
            "verbose": True,
            "compile": False,
            "checkpoint_rate": 10,
            "use_val": False,
            "train_size": 0.8,
            "test_size": 0.2,
            "val_size": 0.0,
            "test_rate": None,
            "validation_rate": None,
            "benchmark_rate": None,
            "train_epochs": 12000,
            "warmup_steps": 150,
            "shuffle_data": False,
            "new": True,
            "time_precision": "ms",
            "val_progress_config": progress_base,
            "test_progress_config": progress_base,
            "train_progress_config": progress_base
        }
        defined = []
        values = {}
        for i, (param, value) in enumerate(config.items()):
            if param not in params:
                raise KeyError("Argument defined in JSON file not found in available arguments.")
            if param == "optimizer":
                try:
                    exec(f"from torch.optim import {value}")
                except Exception as e:
                    raise e
                else:
                    exec(f"values['{param}'] = {value}")
            elif param == "nlinear":
                try:
                    exec(f"nn.{value}")
                except Exception as e:
                    try:
                        exec(f"from base import {value}")
                    except Exception as e:
                        raise e
                    else:
                        exec(f"values['{param}'] = {value}")
                else:
                    exec(f"values['{param}'] = nn.{value}")
            elif param == "attention_method":
                if value in (names := [att.__name__ for att in attention_types]):
                    exec(f"from .transformer.attention import {value}")
                    exec(f"values['{param}'] = {value}")
                elif value is None:
                    values[param] = MultiHeadAttention
                else:
                    raise TypeError(f"The requested attention method ({value}) is not one of the available types ({names}).")
            else:
                # print(param, value, params.index(param), defaults[params.index(param)], default_types[params.index(param)])
                if not issubclass(value.__class__, default_types[param]):
                    raise TypeError(f"The type of argument '{param}' conficts with the expected type ({type(default_types[param])}).")
                values[param] = value
            defined.append(param)
        for param in set(tuple(params)).difference(set(tuple(defined))):
            if param == "attention_method":
                values[param] = MultiHeadAttention
            else:
                values[param] = defaults[param]

        return cls(**values)
