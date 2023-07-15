# Fractals

This package contains code for sampling and rendering Iterated Function Systems and diamond-square textures.
The implementation uses `numba` extensively for performance, as well as `numpy` and `opencv` (for colorspace conversion).

## Sampling IFS Codes

You can generate a dataset of fractal systems saved as a `pickle` file by running `ifs.py`:
```bash
# generate a dataset of 50,000 systems, each with between 2 and 4 affine transformations
python ifs.py --save_path ifs-50k.pkl --num_systems 50000 --min_n 2 --max_n 4
```
This will produce a `.pkl` file containing a dictionary with the following structure:
```python
{
  "params": [
    {"system": np.array(...)},
    ...
  ],
  "hparams": {
    ...
  }
}
```

Or you can use `ifs` as a library:
```python
from fractal_learning.fractals import ifs

system = ifs.sample_system(2)
```

## Rendering Images

```python
from fractal_learning.fractals import ifs, diamondsquare

system = ifs.sample_system(2)
points = ifs.iterate(sys, 100000)

# render images in binary, grayscale, and color
binary_image = ifs.render(points, binary=True)
gray_image = ifs.render(points, binary=False)
color_image = ifs.colorize(gray_image)

# create a random colored background
background = diamondsquare.colorized_ds()

# create a composite image
composite = background.copy()
composite[gray_image.nonzero()] = color_image[gray_image.nonzero()]
```
