from abc import abstractmethod, ABC
from typing import Literal, Tuple
import numpy as np
import torch

class BaseDataLoader(ABC):
    def __init__(self,
                 batch_size: int,
                 context_size: int,
                 epoch_batch_size: int,
                 split: Literal['val', 'train'],
                 rank: int,
                 world_size: int):
        self.B, self.T, self.w_size = batch_size, context_size, world_size
        self.rank = rank
        self.current_position = self.B * self.T * self.rank
        self.iterations = 0
        self.current_shard = 0
        self.split = split
        assert epoch_batch_size % (self.B * self.T * self.w_size) == 0, "Cannot divide the processed tokens per epoch into an integer number of gradient accumulation steps."
        self.limit = epoch_batch_size // (self.B * self.T * self.w_size)

    @staticmethod
    def load_tokens(filename: str, path: str = ""):
        npt: np.ndarray = np.load(path + filename)
        ptt = torch.tensor(npt.astype(np.int64), dtype=torch.long)
        return ptt


    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Tuple[torch.Tensor]:
        """
        Iterates through the data to feed to the model.
        Returns the inputs to the model in the first positions and the expected outputs of the model in the last position.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Resets the dataloader to an initial state"""
        pass

    @abstractmethod
    def resume(self, state: list) -> None:
        """Sets the dataloader to the state of a previous run"""
        pass
