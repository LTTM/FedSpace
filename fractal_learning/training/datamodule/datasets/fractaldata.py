from functools import partial
import pickle
from typing import Callable, Optional, Tuple, Union
import warnings

from cv2 import GaussianBlur
import numpy as np
import torch
import torchvision

from fractal_learning.fractals import diamondsquare, ifs
from .generator import IFSGenerator, MultiGenerator, SelfSupervisedGenerator


class FractalClassDataset(object):
    def __init__(
        self,
        param_file: str,
        num_systems: int = 1000,
        num_class: int = 1000,
        per_class: int = 100,
        generator: Optional[Callable] = None,
        cache_size: int = 0,
    ):
        self.num_systems = num_systems
        self.num_class = num_class
        self.per_class = per_class
        self.per_system = num_class * per_class / num_systems
        self.params = pickle.load(open(param_file, 'rb'))['params'][:num_systems]

        self.generator = generator or IFSGenerator()

        self.cache_size = cache_size
        self.cache = None
        if cache_size > 0:
            self.cache = []
            T = torchvision.transforms
            self.cache_tform = torchvision.transforms.Compose([
                T.RandomAffine(10, (0.1, 0.1), (0.7, 0.95), interpolation=T.functional.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip()
            ])

    def get_label(self, idx):
        return int(idx // self.num_class)

    def get_system(self, idx):
        return int(idx // self.per_system)

    def get_img(self, idx):
        sysidx = self.get_system(idx)
        img = self.render_img(sysidx)
        img = torch.from_numpy(img).float().mul_(1/255.).permute(2,0,1)
        label = self.get_label(idx)
        return img, label

    def render_img(self, sysidx):
        params = self.params[sysidx]['system']
        img = self.generator(params)
        return img

    def __len__(self):
        return self.num_class * self.per_class

    def __getitem__(self, idx):
        if self.cache is None or len(self.cache) < self.cache_size:
            img, label = self.get_img(idx)
            if self.cache is not None:
                self.cache.append((img, label))
        elif np.random.default_rng().random() < 0.5:
            img, label = self.get_img(idx) 
            self.cache[idx % self.cache_size] = (img, label)
        else:
            img, label = self.cache[idx % self.cache_size]
            img = self.cache_tform(img)

        return img, label


class MultiFractalDataset(object):
    def __init__(
        self,
        param_file: str,
        num_systems: int = 1000,
        num_class: int = 1000,
        per_class: int = 1000,
        generator: Optional[Callable] = None,
        period: int = 2,
    ):
        self.num_systems = num_systems
        self.num_class = num_class
        self.per_class = per_class
        self.per_system = num_class * per_class / num_systems
        self.params = pickle.load(open(param_file, 'rb'))['params'][:num_systems]

        self.generator = generator or MultiGenerator()
        # start with an image in the cache
        k = np.random.default_rng().integers(0, num_class)
        self.generator.add_sample(self.params[k]['system'], label=k)

        self.steps = 0
        self.period = period

    def __len__(self):
        return self.num_class * self.per_class

    def get_label(self, idx):
        return int(idx // self.num_class)

    def get_system(self, idx):
        return int(idx // self.per_system)

    def __getitem__(self, idx):
        # whether it's time to render a new fractal or not
        self.steps = (self.steps + 1) % self.period
        sample = self.steps == 0
        sysidx = self.get_system(idx)
        label = self.get_label(idx)
        params = self.params[sysidx]['system']
        img, labels = self.generator(params, label=label, new_sample=sample)
        img = torch.from_numpy(img).float().mul_(1/255.).permute(2,0,1)

        return img, labels
        

class FractalUnsupervisedDataset(object):
    def __init__(
        self,
        param_file: str,
        num_systems: int = 1000000,
        per_system: int = 1,
        generator: Optional[Callable] = None,
        period: int = 2,
    ):
        self.num_systems = num_systems
        self.per_system = per_system
        self.params = pickle.load(open(param_file, 'rb'))['params'][:num_systems]

        self.generator = generator or SelfSupervisedGenerator()

        self.steps = 0
        self.period = period

    def __len__(self):
        return len(self.params) * self.per_system

    def __getitem__(self, idx):
        # whether it's time to render a new fractal or not
        sample = self.steps == 0
        self.steps = (self.steps + 1) % self.period
        idx = int(idx // self.per_system)
        params = self.params[idx]['system']
        imgs = self.generator(params, new_sample=sample)
        imgs = [torch.from_numpy(img).float().mul_(1/255.).permute(2,0,1) for img in imgs]

        return imgs
