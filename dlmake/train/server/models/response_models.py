from pydantic import BaseModel


class FloatResponseModel(BaseModel):
    value: float


class IntegerResponseModel(BaseModel):
    value: int

class SeqLenResponseModel(BaseModel):
    ctxs: list[int]
    losses: list[float]
