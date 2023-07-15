from typing import Optional, Tuple

from .datasets import medical
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset


__all__ = ['GlaSDataModule']


class GlaSDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        size: int = 448,
        augment: bool = True,
        normalize: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment = augment
        self.normalize = normalize

        self.dims = (3, size, size)
        self.num_class = 2

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def train_dataloader(self):
        if self.data_train is None:
            self.data_train = medical.GlaSDataset(
                root = f'{self.data_dir}',
                transform = 'default' if self.augment else medical.ToTensor()
            )

        return DataLoader(
            dataset = self.data_train,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = True,
            drop_last = True,
        )

    def val_dataloader(self):
        if self.data_val is None:
            self.data_val = medical.GlaSDataset(
                root = f'{self.data_dir}',
                train = False,
                transform = medical.ToTensor()
            )

        return DataLoader(
            dataset = self.data_val,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = False,
            drop_last = False,
            collate_fn = val_collate_fn,
        )


def val_collate_fn(batch):
    imgs = [x[0] for x in batch]
    masks = [x[1] for x in batch]
    w = max(x.shape[-1] for x in imgs)
    h = max(x.shape[-2] for x in imgs)
    imgs = [torch.nn.functional.pad(x, (0, w-x.shape[-1], 0, h-x.shape[-2])) for x in imgs]
    masks = [torch.nn.functional.pad(x, (0, w-x.shape[-1], 0, h-x.shape[-2])) for x in masks]
    return torch.stack(imgs, 0), torch.stack(masks, 0)