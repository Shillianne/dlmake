# Training Server API

## API Endpoints

### GET `/health`: Returns health check result

**Response format**

- HTTP status code 200
  - Body: `{"status": "ok" }`
  - Explanation: The server is functioning correctly.

### POST `/update`: Updates the values of the training variables

Used internally within the training code to update the values of the training metrics.

**Response format**

- HTTP status code 200

### POST `/loop`: Updates the values of the looping probabilities

Used internally within the training code to update the values of the unnormalized exit probabilities of a looping model.

**Response format**

- HTTP status code 200

### GET `/loss`: Returns the last value for the training loss

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/acc`: Returns the last value for the training accuracy

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/test_loss`: Returns the last value for the testing loss

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/test_acc`: Returns the last value for the testing accuracy

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/val_loss`: Returns the last value for the validation loss

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/val_acc`: Returns the last value for the validation accuracy

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/norm`: Returns the last value for the gradient norm

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/tkps`: Returns the last value or the training tokens per second

Depending on whether there is a value pending to be read, the endpoint will either return the int value or None.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/mfu`: Returns the last value for the model FLOPS utilization (MFU)

Depending on whether there is a value pending to be read, the endpoint will either return the float value or None.

**Response format**

- HTTP status code 200
  - Body: float or None

### GET `/vram`: Returns the last value for the used GPU VRAM

Depending on whether there is a value pending to be read, the endpoint will either return the int value or None.
The value is measured in MiB.

**Response format**

- HTTP status code 200
  - Body: int or None

### GET `/loop`: Returns the last value for the unnormalized exit probabilities for a looping model

The probabilities are stored in a tensor, so to make them able to be sent they are converted into a stream of bytes. On the receiving side, the byte stream should be unpickled and each element of the resulting list should be loaded with torch.

**Response format**

- HTTP status code 200
  - Body: bytes

