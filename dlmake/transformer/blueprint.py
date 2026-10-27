from abc import ABC, abstractmethod, ABCMeta
from typing import Literal, Optional, Type
from dlmake import Config, OuroExitGate
from .blocks import DecoderBlock
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
        def new_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self.apply(self._init_weights)
        attrs["__init__"] = new_init
        return super(TransformerMeta, cls).__new__(cls, clsname, bases, attrs)
            

class Transformer(nn.Module, ABC, metaclass=TransformerMeta):
    def __init__(self, *args, **kwargs):
        super(Transformer, self).__init__()
        self.config: Config = args[0]

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
    def forward(self, *args, start_pos: int = 0) -> torch.Tensor:
        """
        Forward method of the base transformer.
        Arguments:
            - args (tuple[torch.Tensor]): A tuple of input tensors to pass to the transformer.
            - start_pos (int): During training, this parameter is not used and is always 0. During inference,
              this parameter takes the value of the number of tokens processed so far. This is so that in this mode
              the model only takes a single token as input on each generation iteration, reducing memory overhead.
        """
        pass
    
    def configure_parameters(self, weight_decay: float, lr: float, device: str, custom_optim: Optional[Type[torch.optim.Optimizer]] = None) -> torch.optim.Optimizer:
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
        num_params_no_embed = sum(p.numel() for name, p in param_dict.items() if "embedding" not in name)
        print(f"Number of decayed parameter tensors: {len(decay_params)}, with {num_decayed_params:,} parameters.")
        print(f"Number of non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decayed_params:,} parameters.")
        print(f"Number of parameters without embedding layer: {num_params_no_embed:,} parameters.")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        print(f"Using fused AdamW: {use_fused}")
        if self.config.training.use_fused:
            if custom_optim is None:
                assert self.config.training.optimizer != "custom", "A custom optimizer wasn't provided even though the configuration claimed it so."
                optimizer = self.config.training.optimizer(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)  # type: ignore
            else:
                optimizer = custom_optim(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)  # type: ignore
        else:
            if custom_optim is None:
                assert self.config.training.optimizer != "custom", "A custom optimizer wasn't provided even though the configuration claimed it so."
                optimizer = self.config.training.optimizer(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, is_paged=True)  # type: ignore
            else:
                optimizer = custom_optim(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, is_paged=True)  # type: ignore
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


class BaseTransformer(Transformer):
    def __init__(
        self,
        config: Config,
        ff: Optional[nn.Module] = None,
        shallow_ff: Optional[nn.Module] = None,
    ):
        """
        Sample Transformer class that supports all settings.

        Arguments:
            - config (Config): The configuration for the model.
            - ff (Module | None): An optional feed forwawrd network to use in the model.
            - shallow_ff (Module | None): An optional shallow feed forwawrd network to use in the early layers model.

        Although this models supports all model settings of the library, it contains a lot of conditional statements to
        archieve this. While it will most likely work for the particular task that you may want to train it for, it's
        encouraged to write your own subclass of `Transformer` to optimize inference time and support other features not
        present in the library.
        """
        super(BaseTransformer, self).__init__(config)
        self.embed_dim = config.embed_dim

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            dtype=config.dtype
        )
        self.blocks = nn.ModuleList([
            DecoderBlock(config, layer=i, ff=ff, shallow_ff=shallow_ff) for i in range(config.nlayers)
        ])
        self.norm = config.norm(config.embed_dim, dtype=config.dtype)
        self.classification_layer = nn.Linear(
            in_features=config.vocab_size,
            out_features=config.embed_dim,
            bias=False,
            dtype=config.dtype,
            device="meta",
        )
        self.classification_layer.weight = self.embedding.weight
        if config.use_looping:
            assert self.config.looping.q.value is not None
            self.q = self.config.looping.q.value
            self.gate = OuroExitGate(config)
            if self.training:
                self.register_buffer(
                    "unnormalized",
                    torch.zeros(config.looping.n_max_loops, config.batch_size, config.ctx_size, config.embed_dim)
                )

    def forward(self, *args, start_pos: int = 0):
        (x,) = args
        *_, T = x.shape
        mask: Optional[torch.Tensor] = None
        indices = torch.arange(start_pos, start_pos + T).to(x.device)

        if self.training:
            mask = torch.full((T, T), float("-inf"), device=x.device).triu_(1)
        else:
            if start_pos == 0 and T > 1:
                mask = torch.full((T, T), float("-inf"), device=x.device).triu_(1)
            elif start_pos != 0 and T > 1:
                mask = torch.full((T, T), float("-inf"), device=x.device).triu_(1)[start_pos:, :]
            elif T == 1:
                mask = None

        x = self.embedding(x)

        if self.config.use_looping:
            if self.config.looping.looping_behaviour == "full":
                top = None
                min_idx = 0
                max_idx = len(self.blocks)
            else:  # end
                top = self.blocks[0]
                min_idx = 1
                max_idx = len(self.blocks)

            if top:
                x = top(x)
            probs = []
            survivals: list[torch.Tensor] = []
            unnormalized = []
            cumulative: list[torch.Tensor] = []
            looping_output: list[torch.Tensor] = []
            if not self.training:
                z = torch.zeros_like(x, device=x.device)
            for loop in range(self.config.looping.n_max_loops):
                T = x.size(1)
                if mask is not None:
                    mask = mask[:T, :T]
                for block in self.blocks[min_idx:max_idx]:
                    x = block(x, indices, start_pos=start_pos, mask=mask)
                if self.training:
                    looping_output.append(x)

                p: torch.Tensor = self.gate(x)
                probs.append(p)
                s = torch.prod(1 - torch.cat(probs, dim=-1), dim=-1, keepdim=True) if loop != 0 else 1 - p
                if loop == 0:
                    u = p
                elif loop == len(x) - 1:
                    u = torch.clamp(1 - torch.sum(torch.cat(unnormalized, dim=-1), dim=-1, keepdim=True), min=0, max=1)
                else:
                    u = (p * survivals[-1])
                c = u if loop == 0 else u + cumulative[-1]
                survivals.append(s)
                unnormalized.append(u)
                cumulative.append(c)

                if not self.training:
                    looping_mask = c <= self.q  # tokens that will continue to the next loop
                    opposite = torch.logical_not(looping_mask).squeeze(-1)
                    z[:, indices[opposite.view(-1)], :] = x[opposite]  # type: ignore
                    if True not in looping_mask:
                        break  # if there are no tokens with true on the looping mask, it means no tokens will continue to the next loop and we can exit
                    x = x.masked_select(looping_mask).view(1, -1, self.embed_dim)  # we exclude all tokens that don't continue to the next loop
                    indices = indices[looping_mask.view(-1)]
                    probs = [p[looping_mask].view(1, -1, 1) for p in probs]
                    survivals = [s[looping_mask].view(1, -1, 1) for s in survivals]
                    unnormalized = [u[looping_mask].view(1, -1, 1) for u in unnormalized]
                    cumulative = [c[looping_mask].view(1, -1, 1) for c in cumulative]

            if self.training:
                x: torch.Tensor = torch.stack(looping_output, dim=0)  # (num_loops, batch, ctx_size, embed_dim)
                self.unnormalized = torch.stack(unnormalized, dim=0)  # (num_loops, batch, ctx_size, 1)
        else:
            for block in self.blocks:
                x = block(x, indices, start_pos=start_pos, mask=mask)

        x = self.norm(x)
        logits = self.classification_layer(x)

        return logits


class TrainingModel:
    def __init__(self, model: Transformer):
        self.inner: Transformer = model

    def __call__(self, *args) -> torch.Tensor:
        return self.forward(*args)

    def forward(self, *args) -> torch.Tensor:
        return self.inner(*args, start_pos=0)

    def eval(self):
        self.inner.eval()

    def train(self):
        self.inner.train()


class InferenceModel(nn.Module):
    def __init__(self, model: Transformer):
        super(InferenceModel, self).__init__()

        self.model = model
        self.model.eval()
        self.start_pos = 0
        self.gen_count = 0
        self.inputs: list[torch.Tensor] = []
        assert isinstance(self.model.blocks, nn.ModuleList)
        for layer in self.model.blocks:
            assert isinstance(layer, DecoderBlock)
            layer.attn._is_training = False
            layer.attn.create_kv_cache()

    def forward(self, *args) -> torch.Tensor:
        x: torch.Tensor = args[0]
        _, T = x.shape
        if not self.inputs:
            for arg in args:
                self.inputs.append(arg)
        else:
            for i, arg in enumerate(args):
                self.inputs[i] = torch.hstack([self.inputs[i], arg])
        if T > 1:
            args = self.inputs
        out = self.model(start_pos=self.start_pos, *args)
        self.start_pos += T
        return out

    def generate(self, sequence: torch.Tensor, max_tokens: int = 0, choice: Literal["probabilistic", "top"] = "probabilistic"):
        assert max_tokens > 0, f"Can't generate response with a budget of {max_tokens} tokens."
        assert isinstance(self.model.eos_token, int)
    
        sequence = sequence
        while self.gen_count != max_tokens:
            with torch.inference_mode():
                logits = self(sequence)
                l = logits[:, -1, :]
                probs = torch.softmax(l, dim=1)
                _, topk_indeces = torch.topk(probs, 100, dim=-1)
            p = probs.flatten()
            indeces = topk_indeces.flatten()
            rollover = 1
            if choice == "probabilistic":
                token = indeces[p[indeces].multinomial(num_samples=1)]
            elif choice == "top":
                token = indeces[0].expand(1)
            while token.item() > self.model.eos_token:
                if choice == "probabilistic":
                    token = indeces[p[indeces].multinomial(num_samples=1)]
                elif choice == "top":
                    token = indeces[rollover].expand(1)
                    rollover += 1
            yield token
            self.gen_count += 1
            sequence = token.reshape(1, -1)

