from typing import Callable, Optional, Tuple
import os

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .datasets import fractaldata


class FractalClassDataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        size: int = 256,
        data_file: str = None,
        num_systems: int = 1000,
        num_class: int = 1000,
        per_class: int = 1000,
        generator: Optional[Callable] = None,
        normalize: Optional[str] = None,
        cache_size: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.num_systems = num_systems
        self.num_class = num_class
        self.per_class = per_class
        self.generator = generator
        self.normalize = normalize
        self.cache_size = cache_size

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, size, size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.data_file is None:
            self.data_file = os.path.join("fractal_learning", "training", self.data_dir + 'ifs-100k.pkl')
        else:
            self.data_file = os.path.join("fractal_learning", "training", self.data_dir + self.data_file)
        self.data_train = fractaldata.FractalClassDataset(
            param_file=self.data_file,
            num_systems=self.num_systems,
            num_class=self.num_class,
            per_class=self.per_class,
            generator=self.generator,
            cache_size=self.cache_size
        )
        self.data_val = self.data_train
        self.data_test = None

    def train_dataloader(self):
        if self.data_train is None:
            self.setup()
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.data_val is None:
            self.setup()
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )


class MultiLabelFractalDataModule(LightningDataModule):
    """
    """
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        size: int = 256,
        data_file: str = None,
        num_systems: int = 1000,
        num_class: int = 1000,
        per_class: int = 1000,
        generator: Optional[Callable] = None,
        normalize: Optional[str] = None,
        period: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.num_systems = num_systems
        self.num_class = num_class
        self.per_class = per_class
        self.period = period
        self.generator = generator
        self.normalize = normalize

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, size, size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.data_file is None:
            self.data_file = self.data_dir + 'ifs-100k.pkl'
        else:
            self.data_file = self.data_dir + self.data_file
        self.data_train = fractaldata.MultiFractalDataset(
            self.data_file,
            num_systems = self.num_systems,
            num_class = self.num_class,
            per_class = self.per_class,
            generator = self.generator,
            period = self.period,
        )
        self.data_val = self.data_train
        self.data_test = None

    def train_dataloader(self):
        if self.data_train is None:
            self.setup()
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        if self.data_val is None:
            self.setup()
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate,
        )

    @staticmethod
    def collate(batch):
        imgs = torch.stack([torch.as_tensor(x[0]) for x in batch], 0)
        labels = [torch.as_tensor(x[1]) for x in batch]
        lens = (len(x) for x in labels)
        labs = torch.zeros(len(labels), max(lens), dtype=torch.int64)
        for i in range(len(labels)):
            if len(labels[i]):
                labs[i, :len(labels[i])] = labels[i]
                labs[i, len(labels[i]):] = labels[i][-1]
            else:
                labs[i, :] = -1
        return imgs, labs


class SelfSupervisedFractalDataModule(LightningDataModule):
    """
    """
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        size: int = 224,
        data_file: str = None,
        num_systems: int = 1000000,
        per_system: int = 1,
        generator: Optional[Callable] = None,
        normalize: Optional[str] = None,
        period: int = 2,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.num_systems = num_systems
        self.per_system = per_system
        self.period = period
        self.generator = generator
        self.normalize = normalize

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, size, size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.data_file is None:
            self.data_file = self.data_dir + 'ifs-100k.pkl'
        else:
            self.data_file = self.data_dir + self.data_file
        self.data_train = fractaldata.FractalUnsupervisedDataset(
            self.data_file,
            num_systems = self.num_systems,
            per_system = self.per_system,
            generator = self.generator,
            period = self.period,
        )
        self.data_val = self.data_train
        self.data_test = None

    def train_dataloader(self):
        if self.data_train is None:
            self.setup()
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.data_val is None:
            self.setup()
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
        )
