'''Mixing a target and auxiliary dataset into a single dataset.
'''
import random

import torch
import torchvision


class MixupDataset(object):
    def __init__(
        self,
        target_dataset,
        aux_dataset,
        alpha=[2, 0.4],
        label_method='target_smooth',
        **kwargs,
    ):
        self.target_dataset = target_dataset
        self.aux_dataset = aux_dataset
        if isinstance(alpha, (float, int)):
            alpha = [alpha, alpha]
        self.alpha = alpha

        if label_method == 'target_smooth':
            self.combine_labels = self._target_label_smoothing
        elif label_method == 'hard':
            self.combine_labels = self._hard_target
        elif label_method == 'extra_class':
            self.combine_labels = self._extra_class
        elif hasattr(label_method, '__call__'):
            self.combine_labels = label_method
        else:
            raise AttributeError(f'{label_method} is not a recognized method for combining labels')

        if not hasattr(target_dataset, 'num_class'):
            target_dataset.num_class = max(target_dataset.targets) + 1

    def _target_label_smoothing(self, y1, y2, lmda):
        '''Create a smoothed label for the target dataset label distribution. The target
        class gets probability `lmda`, and the remaining (1-`lmda`) is distributed equally
        among the other classes in the target label space. This does not use the auxiliary
        label space at all.
        '''
        n = self.target_dataset.num_class
        lvec = torch.empty(n).fill_((1 - lmda) / (n - 1))
        lvec[y1] = lmda
        return lvec

    def _hard_target(self, y1, y2, lmda):
        return y1
    
    def _extra_class(self, y1, y2, lmda):
        '''Uses an extra class to represent all fractal images, and creates a normal mixup
        label based on lmda.
        '''
        n = self.target_dataset.num_class
        lvec = torch.zeros(n + 1)
        lvec[y1] = lmda
        lvec[-1] = (1 - lmda)
        return lvec

    def __len__(self):
        return len(self.target_dataset)

    def __getitem__(self, idx):
        img, label = self.target_dataset[idx]
        aidx = random.randint(0, len(self.aux_dataset) - 1)
        aimg, alabel = self.aux_dataset[aidx]

        lmda = random.betavariate(*self.alpha)
        mixed = lmda * img + (1 - lmda) * aimg
        label = self.combine_labels(label, alabel, lmda)

        return mixed, label