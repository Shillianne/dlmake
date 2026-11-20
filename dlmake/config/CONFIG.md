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


## Training Configuration settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| tokens_per_epoch | int | 1024 | How many tokens are consumed by the model in a single accumulated forward pass |
| loaded_tokens | int | 4096 | The number of tokens to be loaded into a temporary buffer for memory optimization |
| max_lr | float | 1e-3 | The maximum learning rate during training |
| min_lr | float | 1e-6 | The minimum learning rate during training |
| optimizer | (torch.optim.Optimizer, Literal(custom)) | torch.optim.AdamW | Optimizer used during training. Can be set to custom for using a custom defined one. |
| scheduler | (torch.optim.lr_scheduler.LRScheduler, Literal(custom)) | torch.optim.lr_scheduler.OneCycleLR | Scheduler used during training. Can be set to custom for using a custom defined one. |
| scheduler_options | GeneralParams | - | A setting that allows any variable names and values for the customization of the parameters of the scheduler |
| loss | (nn.Module, Literal(custom)) | nn.CrossEntropyLoss | Loss function used during training. Can be set to custom for using a custom defined one. |
| loss_options | GeneralParams | - | A setting that allows any variable names and values for the customization of the parameters of the loss function |
| accuracy | (nn.Module, Metric, Literal(custom)) | torchmetrics.Accuracy | The accuracy function used during training. Can be set to custom for using a custom defined one. |
| use_fused | bool | True | Whether to use the fused optimizer implementation for faster weight updates |
| compile | bool | False | Whether to compile the model (can fail depending on specific model implementation) |
| gradient_checkpoint | bool | False | Whether to use gradient checkpointing during training |
| checkpoint_rate | int | 10 | How often a checkpoint of the model is saved |
| save_history | bool | False | Whether or not to save a history of some training metrics |
| use_val | bool | False | Whether to use a validation dataset for training |
| train_size | float | 0.8 | The percentage of the full dataset used for training the model |
| test_size | float | 0.2 | The percentage of the full dataset used for testing the model |
| val_size | float | 0.0 | The percentage of the full dataset used for validation |
| test_rate | (int, None) | None | How often to test the model on the testing dataset |
| val_rate | (int, None) | None | How often to test the model on the validation dataset |
| benchmark_rate | (int, None) | None | How often to test the model on a group of benchmarks |
| train_epochs | int | 12000 | How many series of accumulated forward passes are done in total |


### Server Configuration settings


| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| host | str | 127.0.0.1 | The host for the server used during training |
| port | int | 38267 | The port for the server used during training |
| vram_heartbeat | float | 5.0 | The number of seconds between checking VRAM usage during training |


### Logging Configuration settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| verbose | bool | True | Level of logging during training |
| time_precision | Literal(s, ms) | ms | The units for the duration of each model pass |
| metrics | MetricsSettings | - | The settings for the metrics to record |
| benchmarks | BenchmarksSettings | - | The settings for the benchmarks to record |


### Metrics Configuration settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| lr | bool | False | Whether to record learning rate during training |
| norm | bool | False | Whether to record gradient norm during training |
| mfu | bool | False | Whether to record running mfu during training |
| time | bool | False | Whether to record training, testing, validation and benchmarking during training |
| tok | bool | False | Whether to record tokens per second during training |
| eta | bool | False | Whether to record estimated remaining time during training |
| precision | bool | False | Whether to record model precision during training |
| recall | bool | False | Whether to record model recall during training |
| specificity | bool | False | Whether to record model specificity during training |
| f1score | bool | False | Whether to record model F-1 Score during training |
| perplexity | bool | False | Whether to record model perplexity during training |
| calibration_error | bool | False | Whether to record model Calibration Error during training |
| topk_accuracy | bool | False | Whether to record model top-K accuracy during training |
| seqlen_loss | bool | False | Whether to record model's loss for different context lengths during training |


### Benchmarks Configuration settings
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| hellaswag | bool | False | Whether to benchmark the model using the Hellaswag dataset |
| arc_easy | bool | False | Whether to benchmark the model using the ARC-Easy dataset |
| arc_challenge | bool | False | Whether to benchmark the model using the ARC-Challenge dataset |
| piqa | bool | False | Whether to benchmark the model using the PIQA dataset |
| sciq | bool | False | Whether to benchmark the model using the SciQ dataset |
| winogrande | bool | False | Whether to benchmark the model using the Winogrande dataset |
| openbookqa | bool | False | Whether to benchmark the model using the OpenBookQA dataset |
| commonsenseqa | bool | False | Whether to benchmark the model using the CommonSenseQA dataset |
| mmlu | bool | False | Whether to benchmark the model using the MMLU dataset |


## Feed Forward Configuration settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| linear_scale | int | 4 | The scaling to the embedding dimension that happens inside a feed forward layer |
| nlinear | torch.nn.Module | torch.nn.ReLU | The non linear function used in any feed forward layer |
| dropout | float | 0.0 | The dropout probability for the GPTFeedForward network |
| ff | (torch.nn.Module, str) | GPTFeedForward | The torch module for the feed forward network or the sting "custom" for a user-defined one |
| shallow_ff | (torch.nn.Module, str, None) | None | An optional torch module for a different feed forward network on the shallow layers of the model, or the string "custom" for a user-defined one |
| shallow_threshold | int | 100 | How many of the first blocks of the model are going to use the shallow feed forward network |
| bias | bool | True | Specific bias controller for the feed forward networks |


### MoE Configuration settings
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| n_experts | int | 1 | Number of experts in the MoE layer |
| experts_per_tok | int | 1 | Number of contributing experts for each token |
| use_shared | bool | False | Whether to use a shared expert that always gets the input |
| hidden_dim | int | 128 | Hidden dimension between MoE expert layers |
| nlinear | torch.nn.Module | torch.nn.SiLU | Non linear function for the expert network |
| router_scaling | float | 1.0 | Scaling to the contributions of each expert to the tokens |
| score_function | Literal(softmax, sigmoid) | softmax | Function used to obtain the contribution scores for each expert |
| bias | bool | True | Specific bias controller for the MoE gate weights |
| use_latent | bool | False | Whether to use a latent variation of base MoE |
| latent_dim | int | 1 | Latent dimension to use in place of the model dimension in Latent MoE |
| latent_shared | bool | False | Whether to use latent embeddings with the shared expert |
| n_experts | int | 1 | Number of experts in the MoE layer |


Shallow MoE settings follow the same pattern as normal MoE settings since it's just another MoE for lower layers of the model. It can be accessed with with the `shallow_moe` setting.

## Attention Configuration settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| attention.attention_method | torch.nn.Module | MultiHeadAttention | The attention method to use in each attention layer |
| attention.use_rope | bool | False | Whether to use Rotary Positional Embeddings (RoPE) |
| attention.use_qknorm | bool | True | Whether to apply a normalization to the Q and K matrices in each attention layer |
| attention.qk_norm_type | torch.nn.Module | torch.nn.RMSNorm | The type of normalization function to apply to the Q and K matrices |
| attention.n_kv_groups | (int, None) | None | The number of K and V groups to have in Grouped Query Attention |
| attention.q_head_dim | (int, None) | None | The number of embedding dimensions the Q matrix has in MultiHead Latent Attention |
| attention.v_head_dim | (int, None) | None | The number of embedding dimensions the V matrix has in MultiHead Latent Attention |
| attention.kv_lora_dim | (int, None) | None | The number of embedding dimensions of the KV low rank projection space in MultiHead Latent Attention |
| attention.flash | bool | False | Whether to use Flash Attention in each attention layer (not supported in MultiHeadLatentAttention due to its implementation) |
| attention.dropout | float | 0.0 | Specific dropout controller for attention layers (MultiHeadLatentAttention excluded) |
| attention.bias | bool | True | Specific bias controller for weights in attention layers |


## Looping Configuration settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| looping.n_max_loops | int | 1 | Maximum number of loops that can be performed |
| looping.looping_behaviour | Literal(full, end) | full | The behavior of the loops. Can loop all layers (full) or all except the first one (end) |


### Q Configuration settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| value | (float, None) | 0.9 | The value of Q, used for determining when a token stops looping |
| source | Literal(manual, data) | manual | The origin of the value of Q, can be manually set or given by the data. Settings a data given value does nothing so far. |


## Context Compression Configuration settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| factor | int | 2 | The compression factor, i.e., how much the incomming context is compressed |

