from typing import Literal, Tuple, TypeVar, Generic, overload, Optional
from abc import abstractmethod, ABC
from dlmake import Config
import pickle
import torch
import os


class BaseDataLoader(ABC):
    online: bool
    total_tokens: int

    def __init__(
        self,
        config: Config,
        split: Literal["train", "test", "val"],
        min_idx: int,
        max_idx: int,
        rank: int,
        world_size: int,
    ):
        """
        Base class for any dataloader.

        Arguments:
            - config (Config): The configuration for the model to train.
            - split (str): The type of split the dataloader is processing.
            - min_idx (int): Minimum global token index that belongs to the processing split.
            - max_idx (int): Maximum global token index that belongs to the processing split.
            - rank (int): The index for a distributed machine.
            - world_size (int): The number of distributed systems collaborating.

        DLmake expects the dataloaders used for training to be subclasses of this one, so when making your own custom dataloader make sure to use it.
        The internal logic of the dataloader is left open for customization to adapt to all kinds of datasets and loading behaviours, but the training pipeline expects
        each dataloader to output up `Config.training.tokens_per_epoch` tokens per training step. This is done to allow for the customization of a variable number of
        batch accumulation steps. The specifc number of accumulation steps is indirectly controlled via the changing of the model's context size and batch size, together
        with the tokens per epoch. With this, you would be able to put a target number of tokens per training step and, through batch accumulation, the model will process
        this number of tokens even though it wouldn't be able to do so if they were processed at once.

        Another setting your custom dataloader should take into account is `Config.training.loaded_tokens`. This setting allows you to set a maximum number of tokens that
        will be loaded in system memory at any moment. This is done to allow lower-end systems to process larger datasets by continuously shifting the section of the data
        that is currently loaded.
        """
        self.B, self.T, self.w_size = config.batch_size, config.ctx_size, world_size
        self.rank = rank
        self.split = split
        self.min_idx = min_idx
        self.max_idx = max_idx
        assert config.training.tokens_per_epoch % (self.B * self.T * self.w_size) == 0, "Cannot divide the processed tokens per epoch into an integer number of gradient accumulation steps."

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


class OfflineDataLoader(BaseDataLoader):
    online = False
    total_tokens = -1

    def load_tokens(self, min: int, max: int) -> torch.Tensor:
        raise NotImplementedError("Child classes of OfflineDataLoader have to implement this method")

    def __iter__(self):
        return self


class OnlineDataLoader(BaseDataLoader):
    online = True
    total_tokens = -1

    def load_tokens(self, num_tokens: int) -> torch.Tensor:
        raise NotImplementedError("Child classes of OnlineBaseDataLoader have to implement this method")

    def __iter__(self):
        return self


Val = TypeVar("Val", Literal[True], Literal[False])
Online = TypeVar("Online", Literal[True], Literal[False])


class SplitGenerator(Generic[Val, Online]):
    def __init__(
        self,
        use_val: Val,
        dataloader: type[BaseDataLoader],
        online: Online,
        train_size: float = 0.8,
        test_size: float = 0.2,
        val_size: float = 0.0
    ):
        """
        Class to generate cross validation splits for a data assortment.

        Arguments:
            - use_val (bool): Whether to use a validation partition.
            - data_path (str): Path to the directory containing the numpy files of the dataset.
            - train_size (float): Percentage of the dataset to use for training.
            - test_size (float): Percentage of the dataset to use for testing.
            - val_size (float): Percentage of the dataset to use for validation (ignored if use_val is False).
        """
        self.online = online
        self.use_val = use_val
        self.dataloader = dataloader
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
    def generate_splits(
        self: "SplitGenerator[Literal[True], Literal[False]]",
        config: Config,
        rank: int,
        world_size: int,
        data_path: str,
    ) -> tuple[OfflineDataLoader, OfflineDataLoader, OfflineDataLoader]:
        ...

    @overload
    def generate_splits(
        self: "SplitGenerator[Literal[True], Literal[True]]",
        config: Config,
        rank: int,
        world_size: int,
        total_tokens: int,
    ) -> tuple[OnlineDataLoader, OnlineDataLoader, OnlineDataLoader]:
        ...

    @overload
    def generate_splits(
        self: "SplitGenerator[Literal[False], Literal[False]]",
        config: Config,
        rank: int,
        world_size: int,
        data_path: str,
    ) -> tuple[OfflineDataLoader, OfflineDataLoader]:
        ...

    @overload
    def generate_splits(
        self: "SplitGenerator[Literal[False], Literal[True]]",
        config: Config,
        rank: int,
        world_size: int,
        total_tokens: int,
    ) -> tuple[OnlineDataLoader, OnlineDataLoader]:
        ...

    def generate_splits(
        self,
        config: Config,
        rank: int,
        world_size: int,
        *args,
        **kwargs,
    ):
        data_path: Optional[str] = None
        total_tokens = 0
        for k, v in kwargs.items():
            if k == "data_path" and self.online:
                raise TypeError(f"generate_splits() got an unexpected keyword argument '{k}'")
            elif k == "total_tokens" and not self.online:
                raise TypeError(f"generate_splits() got an unexpected keyword argument '{k}'")
            elif k == "data_path" and not self.online:
                data_path = v
            else:
                total_tokens = v
                

        if not self.online and data_path is None:
            raise ValueError(f"An offline dataset requires a path to the data")

        self.testing = int(os.environ.get("DLMAKE_TESTING", -1)) != -1
        if not self.testing:
            if not self.online:
                assert data_path is not None
                with open(os.path.join(data_path, "metadata.pkl"), "rb") as pf:
                    metadata = pickle.load(pf)
                self.total_tokens = metadata["total_tokens"]
            else:
                assert total_tokens > 0, "Cannot build a dataset without any tokens"
                self.total_tokens = total_tokens
        else:
            self.total_tokens = None

        if self.use_val:
            if not self.testing:
                assert self.total_tokens is not None
                min_train_idx = 0
                max_train_idx = int(self.total_tokens * self.train_size)
                min_test_idx = max_train_idx
                max_test_idx = int(self.total_tokens * (self.train_size + self.test_size))
                min_val_idx = max_test_idx
                max_val_idx = self.total_tokens
            else:
                min_train_idx = 0
                max_train_idx = 0
                min_test_idx = 0
                max_test_idx = 0
                min_val_idx = 0
                max_val_idx = 0
            
            return (
                self.dataloader(
                    config=config,
                    split="train",
                    min_idx=min_train_idx,
                    max_idx=max_train_idx,
                    rank=rank,
                    world_size=world_size,
                ),
                self.dataloader(
                    config=config,
                    split="test",
                    min_idx=min_test_idx,
                    max_idx=max_test_idx,
                    rank=rank,
                    world_size=world_size,
                ),
                self.dataloader(
                    config=config,
                    split="val",
                    min_idx=min_val_idx,
                    max_idx=max_val_idx,
                    rank=rank,
                    world_size=world_size,
                ),
            )
        else:
            if not self.testing:
                assert self.total_tokens is not None
                min_train_idx = 0
                max_train_idx = int(self.total_tokens * self.train_size)
                min_test_idx = max_train_idx
                max_test_idx = self.total_tokens
            else:
                min_train_idx = 0
                max_train_idx = 0
                min_test_idx = 0
                max_test_idx = 0

            return (
                self.dataloader(
                    config=config,
                    split="train",
                    min_idx=min_train_idx,
                    max_idx=max_train_idx,
                    rank=rank,
                    world_size=world_size,
                ),
                self.dataloader(
                    config=config,
                    split="test",
                    min_idx=min_test_idx,
                    max_idx=max_test_idx,
                    rank=rank,
                    world_size=world_size,
                ),
            )
