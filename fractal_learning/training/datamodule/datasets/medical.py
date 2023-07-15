from pathlib import Path
import random
from typing import Callable, Optional, Tuple, Union

from PIL import Image
import numpy as np
import torch
import torchvision


class GlaSDataset(object):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Union[Callable, str]] = None,
    ):
        self.root = Path(root)
        self.train = train
        self.transform = transform
        if isinstance(transform, str) and transform == 'default':
            self.transform = self._default_transform()

        prefix = 'train' if train else 'test'

        imgs, annos = [], []
        for fname in self.root.glob(f'{prefix}*.bmp'):
            fname = fname.name
            if 'anno' in fname:
                annos.append(fname)
            else:
                imgs.append(fname)

        imgs.sort(key=lambda x: int(x[:-4].split('_')[1]))
        annos.sort(key=lambda x: int(x[:-4].split('_')[1]))
        self.imgs = imgs
        self.annos = annos

    def _default_transform(self):
        return Compose([
            RandomCrop(),
            RandomFlips(),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.root / self.imgs[idx])
        mask = Image.open(self.root / self.annos[idx])

        if self.transform:
            # transform should modify both the image and the mask
            img, mask = self.transform(img, mask)
        
        return img, mask


class RandomCrop(object):
    def __init__(self, size=448, srange=(0.8, 0.85)):
        self.size = size
        self.srange = srange

    def _sample(self, img_size):
        w, h = img_size
        s = int(random.uniform(*self.srange) * min(w, h))
        x = random.randint(0, w-s-1)
        y = random.randint(0, h-s-1)
        return x, y, s

    def __call__(self, *imgs):
        # images are assumed to be PIL Images of the same size
        x, y, s = self._sample(imgs[0].size)
        imgs = [img.crop((x, y, x+s, y+s)).resize((self.size, self.size), Image.BILINEAR)
                for img in imgs]
        return imgs


class RandomFlips(object):
    def _sample(self):
        if random.random() > 0.5:
            return random.randint(0, 6)
        return -1

    def __call__(self, *imgs):
        code = self._sample()
        if code == -1:
            return imgs
        return [img.transpose(code) for img in imgs]


class ToTensor(object):
    def __call__(self, *imgs):
        img = imgs[0]
        mask = imgs[1]
        return [torchvision.transforms.functional.to_tensor(img),
                torch.as_tensor(np.array(mask), dtype=torch.int64)[None]]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *imgs):
        for tform in self.transforms:
            imgs = tform(*imgs)
        return imgs
