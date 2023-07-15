from cv2 import cvtColor, COLOR_HSV2RGB
import numba
import numpy as np


@numba.njit(cache=True)
def diamond_square(n, decay=0.5, fixed_corners=True):
    s = 2**n + 1
    a = np.zeros((s, s))
    if fixed_corners:
        a[0, 0] = a[0, s-1] = a[s-1, 0] = a[s-1, s-1] = 0.5
    else:
        a[0, 0] = np.random.rand()
        a[0, s-1] = np.random.rand()
        a[s-1, 0] = np.random.rand()
        a[s-1, s-1] = np.random.rand()
    
    for k in range(1, n+1):
        m = 0.5 * np.exp(decay * (1-k))
        ss = s // (2**k)
        
        # diamond
        ni = 2**k
        for i in range(0, ni, 2):
            # s / 2**k
            ru = i * ss
            r = ru + ss
            rd = r + ss
            for j in range(0, ni, 2):
                cl = j * ss
                c = cl + ss
                cr = c + ss
                a[r, c] = 0.25 * (a[ru, cl] + a[ru, cr] + a[rd, cl] + a[rd, cr])
                a[r, c] += np.random.uniform(-m, m)
        
        # square
        ni = 2**k + 1
        for i in range(ni):
            r = i * ss
            if r > 0: ru = r - ss
            else: ru = s - ss - 1
            if r < s-1: rd = r + ss
            else: rd = ss
            sj = 1 if i % 2 == 0 else 0
            for j in range(sj, ni, 2):
                c = j * ss
                if c > 0: cl = c - ss
                else: cl = s - ss - 1
                if c < s-1: cr = c + ss
                else: cr = ss
                a[r, c] = 0.25 * (a[ru, c] + a[r, cl] + a[r, cr] + a[rd, c])
                a[r, c] += np.random.uniform(-m, m)
    return a


@numba.njit(cache=True)
def _colorize(ds):
    img = np.empty((ds.shape[0], ds.shape[1], 3), dtype=np.uint8)

    hue_scale = np.random.uniform(.25, 1) * 255
    hue_shift = np.random.rand() * 255

    sat_scale = np.random.uniform(0.1, 0.3) * 255
    sat_shift = np.random.uniform(0.4, 0.6) * 255

    val_scale = np.random.uniform(0.3, 0.6) * 255
    val_shift = np.random.uniform(0.4, 0.6) * 255

    for i in range(ds.shape[0]):
        for j in range(ds.shape[1]):
            x = ds[i, j]
            img[i, j, 0] = np.uint8(x * hue_scale + hue_shift)  # implicit MOD(256)
            img[i, j, 1] = np.uint8(min(x * sat_scale + sat_shift, 255))
            img[i, j, 2] = np.uint8(min(x * val_scale + val_shift, 255))

    return img


def colorized_ds(size=256):
    n = int(np.ceil(np.log2(size)))
    rng = np.random.default_rng()

    r = diamond_square(n, rng.uniform(0.4, 0.8), fixed_corners=False)[:size, :size]
    
    img = _colorize(r)
    img = cvtColor(img, COLOR_HSV2RGB, dst=img)
    return img