'''Random noise datasets
'''
import random

import torch
import torchvision

from fractal_learning.fractals import diamondsquare


class RandomDataset(object):
    def __init__(
        self,
        noise='white',
        size=224,
        **kwargs,
    ):
        self.size = size
        if noise == 'white':
            self.get_noise = self._white_noise
        elif noise == 'diamondsquare':
            self.get_noise = self._diamondsquare_noise
        else:
            raise ValueError(f'"{noise}" is not a valid noise option.')

    def _white_noise(self):
        return torch.rand(3, self.size, self.size)
    
    def _diamondsquare_noise(self):
        noise = torch.as_tensor(diamondsquare.colorized_ds(self.size))
        noise = noise.permute(2,0,1).float().div_(255).clamp_(0, 1)
        return noise

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.get_noise(), 0