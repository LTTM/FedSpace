from typing import Callable, Optional, Tuple

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .datasets import mixeddata


class MixupDataModule(LightningDataModule):
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
        target_dataset,
        aux_dataset,
        val_dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        size: int = 224,
        alpha: Tuple[float, float] = (2, 0.4),
        label_method: str = 'target_smooth',
        **kwargs,
    ):
        super().__init__()

        self.target_dataset = target_dataset
        self.aux_dataset = aux_dataset
        self.val_dataset = val_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.alpha = alpha
        self.label_method = label_method

        # self.dims is returned when you call datamodule.size()
        self.dims = (3, size, size)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        self.data_train = mixeddata.MixupDataset(
            target_dataset = self.target_dataset,
            aux_dataset = self.aux_dataset,
            alpha = self.alpha,
            label_method = self.label_method,
        )
        self.data_val = self.val_dataset

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
