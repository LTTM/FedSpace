import argparse
from pathlib import Path
import re

import imageio
import numpy as np
from PIL import Image
import torch
import tqdm


def first_layer(filepath, param_name='conv1.weight'):
    x = torch.load(filepath, map_location='cpu')
    if 'state_dict' in x:
        x = x['state_dict']
    try:
        return x[param_name]
    except:
        return x['model.'+param_name]


def to_img(w):
    w = w.numpy().copy()
    w -= w.min()
    w *= (255 / w.max())
    w = w.astype('uint8')
    w = w.reshape(8, 8, 3, 7, 7).transpose(0, 3, 1, 4, 2)
    w = np.pad(w, ((0,0),(0,1),(0,0),(0,1),(0,0)), mode='constant', constant_values=255)
    w = w.reshape(8 * w.shape[1], 8*w.shape[3], 3)
    return w[:-1, :-1]


def create_gif(path, every=1, offset=0, scale=4, pattern='*.ckpt'):
    path = Path(path)
    cps = sorted(path.glob(pattern), key=lambda x: int(re.search(r'(\d+)\.', x.name).groups()[0]))
    with imageio.get_writer(path / f'first_layer_every-{every}.gif', mode='I', duration=0.25) as writer:
        subset = cps[offset::every]
        if (len(cps) - 1) % every != 0:
            subset += [cps[-1]]
        for f in tqdm.tqdm(subset):
            img = Image.fromarray(to_img(first_layer(f)))
            img = img.resize((img.size[0] * scale, img.size[1] * scale), Image.NEAREST)
            writer.append_data(np.asarray(img))


if __name__ == '__main__':
    # add argument parser and call create if
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to directory containing checkpoints')
    parser.add_argument('--every', type=int, default=1, help='Use every nth frame')
    parser.add_argument('--offset', type=int, default=0, help='Start from the kth frame')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor (positive integer) for making a larger image')
    parser.add_argument('-p', '--pattern', type=str, default='*epoch=*.ckpt', help='Glob pattern for checkpoint filenames')
    args = parser.parse_args()
    create_gif(args.path, args.every, args.offset, args.scale, args.pattern)
