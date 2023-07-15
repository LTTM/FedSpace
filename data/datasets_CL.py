import copy
import logging
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR100

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, classes, dataidxs=None, train=True, transform=None, target_transform=None, download=False, bs=None):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.classes = classes
        self.indexes_task = []
        self.bs = bs

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        if self.classes is not None:
            list(map(self.indexes_task.extend, [np.where(target == c)[0].tolist() for c in self.classes]))
            data = data[self.indexes_task]
            target = target[self.indexes_task]

            # permute data
            permutation = np.random.permutation(len(data))
            data = data[permutation]
            target = target[permutation]

            if self.bs is not None and len(data) % self.bs != 0:
                data_tmp = copy.deepcopy(data)
                target_tmp = copy.deepcopy(target)
                filler_perm = np.random.permutation(self.bs - len(data_tmp) % self.bs)
                insuff_data = len(filler_perm) > len(data)
                while insuff_data:
                    permutation = np.random.permutation(len(data))
                    data_tmp = np.concatenate([data_tmp, data[permutation]])
                    target_tmp = np.concatenate([target_tmp, target[permutation]])
                    insuff_data = len(filler_perm) > len(data_tmp)
                data = np.concatenate([data, data_tmp[filler_perm]])
                target = np.concatenate([target, target_tmp[filler_perm]])

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)