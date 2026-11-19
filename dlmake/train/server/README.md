# Training Server API

## API Endpoints

### GET `/health`: Returns health check result

**Response format**

- HTTP status code 200
  - Body: `{"status": "ok" }`
  - Explanation: The server is functioning correctly.

### GET `/metrics/loss`: Returns the last value for the training loss

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/metrics/acc`: Returns the last value for the training accuracy

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/metrics/test_loss`: Returns the last value for the testing loss

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/metrics/test_acc`: Returns the last value for the testing accuracy

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/metrics/val_loss`: Returns the last value for the validation loss

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/metrics/val_acc`: Returns the last value for the validation accuracy

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/metrics/norm`: Returns the last value for the gradient norm

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/metrics/tkps`: Returns the last value or the training tokens per second

Depending on whether there is a value pending to be read, the endpoint will either return the int value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/mfu`: Returns the last value for the model FLOPS utilization (MFU)

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/metrics/vram`: Returns the last value for the used GPU VRAM

Depending on whether there is a value pending to be read, the endpoint will either return the int value or None.
The value is measured in MiB.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/usage`: Returns the last value for the usage of the GPU

Depending on whether there is a value pending to be read, the endpoint will either return the int value or None.
The value is measured in percentage.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/epoch`: Returns the last value for the current training epoch

Depending on whether there is a value pending to be read, the endpoint will either return the int value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/eta`: Returns the last value for the estimation of the reamining training time

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/lr`: Returns the last value for the learning rate

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/train_time`: Returns the last value for the current iteration's time spent training the model

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/test_time`: Returns the last value for the current iteration's time spent testing the model

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/val_time`: Returns the last value for the current iteration's time spent validating the model

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/benchmark_time`: Returns the last value for the current iteration's time spent benchmarking the model

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/precision`: Returns the last value for the training precision

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/recall`: Returns the last value for the training recall

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/specificity`: Returns the last value for the training specificity

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/f1score`: Returns the last value for the training F-1 Score

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/perplexity`: Returns the last value for the training perplexity

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/calibration_error`: Returns the last value for the training Calibration Error

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/test_precision`: Returns the last value for the test precision

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/test_recall`: Returns the last value for the test recall

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/test_specificity`: Returns the last value for the test_specificity

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/test_f1score`: Returns the last value for the test F-1 Score

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/test_perplexity`: Returns the last value for the test perplexity

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/test_calibration_error`: Returns the last value for the test Calibration Error

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/val_precision`: Returns the last value for the validation precision

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/val_recall`: Returns the last value for the validation recall

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/val_specificity`: Returns the last value for the validation specificity

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/val_f1score`: Returns the last value for the validation F-1 Score

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/val_perplexity`: Returns the last value for the validation perplexity

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/val_calibration_error`: Returns the last value for the validation Calibration Error

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/metrics/test_topk_acc`: Returns the last value for the test top-K Accuracy

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

The response is formated as 3 floating values:
    - top-5 Accuracy
    - top-10 Accuracy
    - top-50 Accuracy

- HTTP status code 200
  - Body: int or None

### GET `/metrics/val_topk_acc`: Returns the last value for the validation top-K Accuracy

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

The response is formated as 3 floating values:
    - top-5 Accuracy
    - top-10 Accuracy
    - top-50 Accuracy

- HTTP status code 200
  - Body: int or None

### GET `/metrics/test_seqlen_loss`: Returns the last value for the test Sequence length loss

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.
This endpoint returns the loss with increasing sequence lengths, up until the default context size of the model.

**Response format**

The response is formated as 8 floating values for each of the losses.

- HTTP status code 200
  - Body: list or None

### GET `/metrics/val_seqlen_loss`: Returns the last value for the validation Sequence length loss

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.
This endpoint returns the loss with increasing sequence lengths, up until the default context size of the model.

**Response format**

The response is formated as 8 floating values for each of the losses.

- HTTP status code 200
  - Body: list or None

### GET `/metrics/loop`: Returns the last value for the unnormalized exit probabilities for a looping model

The probabilities are stored in a tensor, so to make them able to be sent they are converted into a stream of bytes. On the receiving side, the byte stream should be unpickled and each element of the resulting list should be loaded with torch.

**Response format**

- HTTP status code 200
  - Body: bytes

### GET `/benchmarks/hellaswag`: Returns the last value for the Hellaswag benchmark

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

It stores 2 floating point numbers, the first being the accuracy for the task and the second one the normalized accuracy.

- HTTP status code 200
  - Body: tuple or None

### GET `/benchmarks/arc_easy`: Returns the last value for the ARC-Easy benchmark

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

It stores 2 floating point numbers, the first being the accuracy for the task and the second one the normalized accuracy.

- HTTP status code 200
  - Body: tuple or None

### GET `/benchmarks/arc_challenge`: Returns the last value for the ARC-Challenge benchmark

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

It stores 2 floating point numbers, the first being the accuracy for the task and the second one the normalized accuracy.

- HTTP status code 200
  - Body: tuple or None

### GET `/benchmarks/piqa`: Returns the last value for the PIQA benchmark

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

It stores 2 floating point numbers, the first being the accuracy for the task and the second one the normalized accuracy.

- HTTP status code 200
  - Body: tuple or None

### GET `/benchmarks/sciq`: Returns the last value for the SciQ benchmark

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

It stores 2 floating point numbers, the first being the accuracy for the task and the second one the normalized accuracy.

- HTTP status code 200
  - Body: tuple or None

### GET `/benchmarks/winogrande`: Returns the last value for the Winogrande benchmark

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

It stores 2 floating point numbers, the first being the accuracy for the task and the second one the normalized accuracy.

- HTTP status code 200
  - Body: tuple or None

### GET `/benchmarks/openbookqa`: Returns the last value for the OpenBookQA benchmark

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

It stores 2 floating point numbers, the first being the accuracy for the task and the second one the normalized accuracy.

- HTTP status code 200
  - Body: tuple or None

### GET `/benchmarks/commonsenseqa`: Returns the last value for the CommonSenseQA benchmark

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

It stores 2 floating point numbers, the first being the accuracy for the task and the second one the normalized accuracy.

- HTTP status code 200
  - Body: tuple or None

### GET `/benchmarks/mmlu`: Returns the last value for the MMLU benchmark

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

It stores 2 floating point numbers, the first being the accuracy for the task and the second one the normalized accuracy.

- HTTP status code 200
  - Body: tuple or None
