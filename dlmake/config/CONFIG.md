# Configuration settings for dlmake


| Variable | Type | Default | Description |
|----------|------|---------|-------------|
|   name   | str  | model   | The name of the model |
| ctx_size | int | 64 | The context window of the model |
| max_ctx_size | int | 64 | The maximum context size of the model |
| vocab_size | int | 256 | The vocabulary size of the model |
| embed_dim | int | 1 | Embedding dimension of the model |
| nhead | int | 1 | Number of heads per attention layer |
| nlayers | int | 1 | Number of transformer blocks of the model |
| batch_size | int | 1 | Number of simultaneous token sequences being processed |
| max_batch_size | int | 1 | Maximum batch size of the model |
| bias | bool | True | General bias for the model |
| dtype | torch.dtype | torch.bfloat16 | Initialization type for the model's weights |
| norm | torch.nn.Module | torch.nn.LayerNorm | Normalization class for the model |
| prenorm | bool | True | Whether to apply norm before attention and feed forward layers |
| postnorm | bool | False | Whether to apply norm after attention and feed forward layers (inside residual stream) |
| use_looping | bool | False | Whether to use a Looping Transformer architecture |
| training.tokens_per_epoch | int | 1024 | How many tokens are consumed by the model in a single accumulated forward pass |
| training.max_lr | float | 1e-3 | The maximum learning rate during training |
| training.min_lr | float | 1e-6 | The minimum learning rate during training |
| training.optimizer | torch.optim.Optimizer | torch.optim.AdamW | Optimizer used during training |
| training.use_fused | bool | True | Whether to use the fused optimizer implementation for faster weight updates |
| training.compile | bool | False | Whether to compile the model (can fail depending on specific model implementation) |
| training.checkpoint_rate | int | 10 | How often a checkpoint of the model is saved |
| training.save_history | bool | False | Whether or not to save a history of some training metrics |
| training.use_val | bool | False | Whether to use a validation dataset for training |
| training.train_size | float | 0.8 | The percentage of the full dataset used for training the model |
| training.test_size | float | 0.2 | The percentage of the full dataset used for testing the model |
| training.val_size | float | 0.0 | The percentage of the full dataset used for validation |
| training.test_rate | (int, None) | None | How often to test the model on the testing dataset |
| training.val_rate | (int, None) | None | How often to test the model on the validation dataset |
| training.benchmark_rate | (int, None) | None | How often to test the model on a group of benchmarks |
| training.train_epochs | int | 12000 | How many series of accumulated forward passes are done in total |
| trainng.warmup_steps | int | 150 | How many steps of training are dedicated to increasing the learning rate |
| training.shuffle_data | bool | False | Whether to shuffle the training samples after every loop |
| training.new | bool | True | Whether this is a new model (for restarting stopped training runs) |
| training.logging.verbose | bool | True | Level of logging during training |
| training.logging.time_precision | Literal(s, ms) | ms | The units for the duration of each model pass |
| training.logging.val_progress_config | ProgressConfig | - | Configuration for what stats to show in each epoch log of validation |
| training.logging.test_progress_config | ProgressConfig | - | Configuration for what stats to show in each epoch log of testing |
| training.logging.train_progress_config | ProgressConfig | - | Configuration for what stats to show in each epoch log of training |
| feed_forward.linear_scale | int | 4 | The scaling to the embedding dimension that happens inside a feed forward layer |
| feed_forward.nlinear | torch.nn.Module | torch.nn.ReLU | The non linear function used in any feed forward layer |
| feed_forward.dropout | float | 0.0 | The dropout probability for the GPTFeedForward network |
| feed_forward.ff | (torch.nn.Module, str) | GPTFeedForward | The torch module for the feed forward network or the sting "custom" for a user-defined one |
| feed_forward.shallow_ff | (torch.nn.Module, str, None) | None | An optional torch module for a different feed forward network on the shallow layers of the model, or the string "custom" for a user-defined one |
| feed_forward.shallow_threshold | int | 100 | How many of the first blocks of the model are going to use the shallow feed forward network |
| feed_forward.bias | bool | True | Specific bias controller for the feed forward networks |
| feed_forward.moe.n_experts | int | 1 | Number of experts in the MoE layer |
| feed_forward.moe.experts_per_tok | int | 1 | Number of contributing experts for each token |
| feed_forward.moe.use_shared | bool | False | Whether to use a shared expert that always gets the input |
| feed_forward.moe.hidden_dim | int | 128 | Hidden dimension between MoE expert layers |
| feed_forward.moe.nlinear | torch.nn.Module | torch.nn.SiLU | Non linear function for the expert network |
| feed_forward.moe.router_scaling | float | 1.0 | Scaling to the contributions of each expert to the tokens |
| feed_forward.moe.score_function | Literal(softmax, sigmoid) | softmax | Function used to obtain the contribution scores for each expert |
| feed_forward.moe.bias | bool | True | Specific bias controller for the MoE gate weights |
| feed_forward.moe.use_latent | bool | False | Whether to use a latent variation of base MoE |
| feed_forward.moe.latent_dim | int | 1 | Latent dimension to use in place of the model dimension in Latent MoE |
| feed_forward.moe.latent_shared | bool | False | Whether to use latent embeddings with the shared expert |
| feed_forward.shallow_moe.n_experts | int | 1 | Number of experts in the MoE layer |
| feed_forward.shallow_moe.experts_per_tok | int | 1 | Number of contributing experts for each token |
| feed_forward.shallow_moe.use_shared | bool | False | Whether to use a shared expert that always gets the input |
| feed_forward.shallow_moe.hidden_dim | int | 128 | Hidden dimension between MoE expert layers |
| feed_forward.shallow_moe.nlinear | torch.nn.Module | torch.nn.SiLU | Non linear function for the expert network |
| feed_forward.shallow_moe.router_scaling | float | 1.0 | Scaling to the contributions of each expert to the tokens |
| feed_forward.shallow_moe.score_function | Literal(softmax, sigmoid) | softmax | Function used to obtain the contribution scores for each expert |
| feed_forward.shallow_moe.bias | bool | True | Specific bias controller for the MoE gate weights |
| feed_forward.shallow_moe.use_latent | bool | False | Whether to use a latent variation of base MoE |
| feed_forward.shallow_moe.latent_dim | int | 1 | Latent dimension to use in place of the model dimension in Latent MoE |
| feed_forward.shallow_moe.latent_shared | bool | False | Whether to use latent embeddings with the shared expert |
| attention.attention_method | torch.nn.Module | MultiHeadAttention | The attention method to use in each attention layer |
| attention.use_rope | bool | False | Whether to use Rotary Positional Embeddings (RoPE) |
| attention.use_qknorm | bool | True | Whether to apply a normalization to the Q and K matrices in each attention layer |
| attention.qk_norm_type | torch.nn.Module | torch.nn.RMSNorm | The type of normalization function to apply to the Q and K matrices |
| attention.n_kv_groups | (int, None) | None | The number of K and V groups to have in Grouped Query Attention |
| attention.q_head_dim | (int, None) | None | The number of embedding dimensions the Q matrix has in MultiHead Latent Attention |
| attention.v_head_dim | (int, None) | None | The number of embedding dimensions the V matrix has in MultiHead Latent Attention |
| attention.kv_lora_dim | (int, None) | None | The number of embedding dimensions of the KV low rank projection space in MultiHead Latent Attention |
| attention.flash | bool | False | Whether to use Flash Attention in each attention layer (not supported in MultiHeadLatentAttention due to its implementation) |
| attention.doc_aware | bool | False | Whether to use a Document Aware causal mask that restricts attention between tokens of the same document |
| attention.dropout | float | 0.0 | Specific dropout controller for attention layers (MultiHeadLatentAttention excluded) |
| attention.bias | bool | True | Specific bias controller for weights in attention layers |
| looping.n_max_loops | int | 1 | Maximum number of loops that can be performed |
| looping.looping_behaviour | Literal(full, end) | full | The behavior of the loops. Can loop all layers (full) or all except the first one (end) |
| looping.q.value | (float, None) | 0.9 | The value of Q, used for determining when a token stops looping |
| looping.q.source | Literal(manual, data) | manual | The origin of the value of Q, can be manually set of given by the data |


## ProgressConfig configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| train_progress | bool | False | Whether to display the epoch number of the currently trained model |
| model_progress | bool | False | Whether to display the global epoch number for all models |
| lr | bool | False | Whether to display the learning rate for this epoch |
| norm | bool | False | Whether to display the weight norm for the current epoch |
| mfu | bool | False | Whether to display the Model FLOPS Utilization (MFU) for the current epoch |
| time | bool | False | Whether to display the time taken for the current epoch |
| tok | bool | False | Whether to display the number of tokens processed this epoch |
| eta | bool | False | Whether to display the remaining estimated time for the training of this model |

