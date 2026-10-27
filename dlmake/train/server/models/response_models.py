from pydantic import BaseModel


class FloatResponseModel(BaseModel):
    value: float


class IntegerResponseModel(BaseModel):
    value: int
