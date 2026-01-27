from abc import abstractmethod, ABC
from typing import Literal, Tuple
import numpy as np
import torch
import os

class BaseDataLoader(ABC):
    def __init__(
        self,
        batch_size: int,
        context_size: int,
        max_gb_loaded: int,
        tokens_per_epoch: int,
        split: Literal["train", "test", "val"],
        min_idx: int,
        max_idx: int,
        rank: int,
        world_size: int,
        shuffle: bool,
        token_size: int = 16,
    ):
        """
        Base class for any dataloader.

        Arguments:
            - batch_size (int): The amount of simultaneous context windows to process.
            - context_size (int): The size of the context window of the model.
            - tokens_per_epoch (int): The amount of tokens to process per training pass. Doesn't have to be equal to the batch_size * context_size,
              but it has to be divisible by it. If tokens_per_epoch > batch_size * context_size, the requested tokens will be processed in batches and
              their results aggregated at the end.
            - split (str): The type of split the dataloader is processing.
            - min_idx (int): Minimum global token index that belongs to the processing split.
            - max_idx (int): Maximum global token index that belongs to the processing split.
            - rank (int): The index for a distributed machine.
            - world_size (int): The number of distributed systems collaborating.
        """
        self.B, self.T, self.w_size = batch_size, context_size, world_size
        self.rank = rank
        self.current_position = self.B * self.T * self.rank
        self.iterations = 0
        self.current_shard = 0
        self.shuffle = shuffle
        self.split = split
        self.min_idx = min_idx
        self.max_idx = max_idx
        self.num_tokens = int((max_gb_loaded * (8 * (1024 ** 3))) / token_size)
        self.min_loaded_idx = 0
        self.max_loaded_idx = self.num_tokens
        assert tokens_per_epoch % (self.B * self.T * self.w_size) == 0, "Cannot divide the processed tokens per epoch into an integer number of gradient accumulation steps."
        self.limit = tokens_per_epoch // (self.B * self.T * self.w_size)

    def load_tokens(self, filename: str, path: str = ""):
        npt: np.ndarray = np.load(path + filename, mmap_mode="r")
        ptt = torch.tensor(npt[self.min_loaded_idx:self.max_loaded_idx])
        return ptt


    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Tuple[torch.Tensor, ...]:
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


class SplitGenerator:
    def __init__(
        self,
        data_path: str,
        shard_size: int,
        use_val: bool,
        train_size: float = 0.8,
        test_size: float = 0.2,
        val_size: float = 0.0
    ):
        """
        Class to generate cross validation splits for a data assortment.

        Arguments:
            - data_path (str): Path to the directory containing the numpy files of the dataset.
            - shard_size (int): Size of each data shard in tokens.
        """
        shards = len(os.listdir(data_path))
        self.total_tokens = shards * shard_size
        self.data_path = data_path
        self.use_val = use_val
        if use_val:
            assert val_size != 0 and train_size + test_size + val_size == 1, "All partitions usage must add to 1."
            self.train_size = train_size
            self.test_size = test_size
            self.val_size = val_size
        else:
            assert train_size + test_size == 1, "All partitions usage must add to 1."
            self.train_size = train_size
            self.test_size = test_size
            self.val_size = 0.0

    def generate_splits(self, dataloader: type[BaseDataLoader], **kwargs):
        batch_size = kwargs["batch_size"]
        context_size = kwargs["context_size"]
        tokens_per_epoch = kwargs["tokens_per_epoch"]
        rank = kwargs["rank"]
        world_size = kwargs["world_size"]
        if self.use_val:
            min_train_idx = 0
            max_train_idx = int(self.total_tokens * self.train_size)
            min_test_idx = max_train_idx
            max_test_idx = int(self.total_tokens * (self.train_size + self.test_size))
            min_val_idx = max_test_idx
            max_val_idx = self.total_tokens
            
            return (dataloader(batch_size, context_size, tokens_per_epoch, split="train", min_idx=min_train_idx, max_idx=max_train_idx, rank=rank, world_size=world_size, shuffle=True),
                    dataloader(batch_size, context_size, tokens_per_epoch, split="test", min_idx=min_test_idx, max_idx=max_test_idx, rank=rank, world_size=world_size, shuffle=True),
                    dataloader(batch_size, context_size, tokens_per_epoch, split="val", min_idx=min_val_idx, max_idx=max_val_idx, rank=rank, world_size=world_size, shuffle=True))
        else:
            min_train_idx = 0
            max_train_idx = int(self.total_tokens * self.train_size)
            min_test_idx = max_train_idx
            max_test_idx = self.total_tokens
            return (dataloader(batch_size, context_size, tokens_per_epoch, split="train", min_idx=min_train_idx, max_idx=max_train_idx, rank=rank, world_size=world_size, shuffle=True),
                    dataloader(batch_size, context_size, tokens_per_epoch, split="test", min_idx=min_test_idx, max_idx=max_test_idx, rank=rank, world_size=world_size, shuffle=True))

