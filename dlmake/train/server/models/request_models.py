from pydantic import BaseModel
from typing import Optional


class UpdateRequestModel(BaseModel):
    loss: Optional[float]
    acc: Optional[list[float]]
    test_loss: Optional[float]
    test_acc: Optional[list[float]]
    val_loss: Optional[float]
    val_acc: Optional[list[float]]
    norm: Optional[float]
    tkps: Optional[float]
    mfu: Optional[float]
    vram_usage: Optional[int]
    epoch: Optional[int]
    eta: Optional[float]
