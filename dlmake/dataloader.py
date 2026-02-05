from typing import Literal, Tuple, TypeVar, Generic, overload
from abc import abstractmethod, ABC
import pickle
import torch
import os


class BaseDataLoader(ABC):
    def __init__(
        self,
        batch_size: int,
        context_size: int,
        loaded_tokens: int,
        tokens_per_epoch: int,
        split: Literal["train", "test", "val"],
        min_idx: int,
        max_idx: int,
        rank: int,
        world_size: int,
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
        self.split = split
        self.min_idx = min_idx
        self.max_idx = max_idx
        assert tokens_per_epoch % (self.B * self.T * self.w_size) == 0, "Cannot divide the processed tokens per epoch into an integer number of gradient accumulation steps."

    def load_tokens(self, min: int, max: int) -> torch.Tensor:
        raise NotImplementedError("Child classes of BaseDataLoader have to implement this method")

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
    def resume(self, **state) -> None:
        """Sets the dataloader to the state of a previous run"""
        pass


Val = TypeVar("Val", Literal[True], Literal[False])
DataLoaderType = TypeVar("DataLoaderType", bound=BaseDataLoader)


class SplitGenerator(Generic[Val]):
    def __init__(
        self,
        data_path: str,
        use_val: Val,
        train_size: float = 0.8,
        test_size: float = 0.2,
        val_size: float = 0.0
    ):
        """
        Class to generate cross validation splits for a data assortment.

        Arguments:
            - data_path (str): Path to the directory containing the numpy files of the dataset.
            - use_val (bool): Whether to use a validation partition.
            - train_size (float): Percentage of the dataset to use for training.
            - test_size (float): Percentage of the dataset to use for testing.
            - val_size (float): Percentage of the dataset to use for validation (ignored if use_val is False).
        """
        with open(os.path.join(data_path, "metadata.pkl"), "rb") as pf:
            metadata = pickle.load(pf)
        self.total_tokens = metadata["total_tokens"]
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

    @overload
    def generate_splits(self: "SplitGenerator[Literal[True]]", dataloader: type[DataLoaderType], **kwargs) -> tuple[DataLoaderType, DataLoaderType, DataLoaderType]:
        ...

    @overload
    def generate_splits(self: "SplitGenerator[Literal[False]]", dataloader: type[DataLoaderType], **kwargs) -> tuple[DataLoaderType, DataLoaderType]:
        ...

    def generate_splits(self, dataloader: type[DataLoaderType], **kwargs):
        batch_size: int = kwargs["batch_size"]
        context_size: int = kwargs["context_size"]
        loaded_tokens: int = kwargs["loaded_tokens"]
        tokens_per_epoch: int = kwargs["tokens_per_epoch"]
        rank: int = kwargs["rank"]
        world_size: int = kwargs["world_size"]
        if self.use_val:
            min_train_idx = 0
            max_train_idx = int(self.total_tokens * self.train_size)
            min_test_idx = max_train_idx
            max_test_idx = int(self.total_tokens * (self.train_size + self.test_size))
            min_val_idx = max_test_idx
            max_val_idx = self.total_tokens
            
            return (
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="train",
                    min_idx=min_train_idx,
                    max_idx=max_train_idx,
                    rank=rank,
                    world_size=world_size,
                ),
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="test",
                    min_idx=min_test_idx,
                    max_idx=max_test_idx,
                    rank=rank,
                    world_size=world_size,
                ),
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="val",
                    min_idx=min_val_idx,
                    max_idx=max_val_idx,
                    rank=rank,
                    world_size=world_size,
                ),
            )
        else:
            min_train_idx = 0
            max_train_idx = int(self.total_tokens * self.train_size)
            min_test_idx = max_train_idx
            max_test_idx = self.total_tokens
            return (
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="train",
                    min_idx=min_train_idx,
                    max_idx=max_train_idx,
                    rank=rank,
                    world_size=world_size,
                ),
                dataloader(
                    batch_size=batch_size,
                    context_size=context_size,
                    loaded_tokens=loaded_tokens,
                    tokens_per_epoch=tokens_per_epoch,
                    split="test",
                    min_idx=min_test_idx,
                    max_idx=max_test_idx,
                    rank=rank,
                    world_size=world_size,
                ),
            )


if __name__ == "__main__":
    class ExampleDataLoader(BaseDataLoader):
        def __init__(
            self,
            batch_size: int,
            context_size: int,
            loaded_tokens: int,
            tokens_per_epoch: int,
            split: Literal["train", "test", "val"],
            min_idx: int,
            max_idx: int,
            rank: int,
            world_size: int
        ):
            super().__init__(
                batch_size,
                context_size,
                loaded_tokens,
                tokens_per_epoch,
                split,
                min_idx,
                max_idx,
                rank,
                world_size
            )
            self.example_attr = 0

    (train_dataloader, test_dataloader) = SplitGenerator(
        data_path="",
        use_val=False
    ).generate_splits(
        dataloader=ExampleDataLoader
    )
    e = train_dataloader.example_attr
