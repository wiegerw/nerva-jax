# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""A synthetic image classification dataset that is generated on the fly.

Each example is a 28x28 grayscale image that contains one of ten shapes: a disk, a ring, a filled square,
a square outline, a triangle, a cross, a plus, horizontal bars, vertical bars or an ellipse. The position,
size, stroke width, rotation and brightness of the shapes are random, and noise is added to the pixels.
The images are flattened to vectors of 784 elements with values in [0, 1], and the targets are class
indices in the range 0-9. This is the same format as the flattened MNIST dataset, so models for MNIST
can be used without changes.

The dataset needs nothing but numpy, and it is deterministic for a given seed. Generating 60000 images
takes a few seconds. The task is easy enough for a small MLP, but not trivial: a linear model reaches
about 40% test accuracy, while an MLP with two hidden layers reaches about 85% after 10 epochs, with a
clear gap between training and test accuracy.

The dataset can also be saved to a .npz file, for example for the mlp command line tool:

    python -m nerva_jax.shapes_dataset --output shapes.npz
"""

import argparse
from typing import Dict, Optional, Tuple

import numpy as np

SHAPE_CLASSES = ('disk', 'ring', 'square', 'square outline', 'triangle', 'cross', 'plus',
                 'horizontal bars', 'vertical bars', 'ellipse')


def _draw_shape(k: int, x: np.ndarray, y: np.ndarray, r: np.ndarray, w: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Draw shape k, given the pixel coordinates (x, y) relative to the centers, the radii r, the stroke widths w,
    and the rotations a."""
    u = x * np.cos(a) + y * np.sin(a)        # rotated coordinates
    v = -x * np.sin(a) + y * np.cos(a)
    if k == 0:    # disk
        mask = u * u + v * v <= r * r
    elif k == 1:  # ring
        mask = np.abs(np.sqrt(u * u + v * v) - r) <= w
    elif k == 2:  # square
        mask = np.maximum(np.abs(u), np.abs(v)) <= r
    elif k == 3:  # square outline
        mask = np.abs(np.maximum(np.abs(u), np.abs(v)) - r) <= w
    elif k == 4:  # triangle
        mask = (v <= 0.6 * r) & (np.abs(u) <= 0.6 * (v + r))
    elif k == 5:  # cross
        mask = (np.minimum(np.abs(u - v), np.abs(u + v)) <= 1.4 * w) & (np.maximum(np.abs(u), np.abs(v)) <= r)
    elif k == 6:  # plus
        mask = (np.minimum(np.abs(u), np.abs(v)) <= w) & (np.maximum(np.abs(u), np.abs(v)) <= r)
    elif k == 7:  # horizontal bars
        mask = (np.abs((v + r) % (0.7 * r) - 0.35 * r) <= 0.6 * w) & (np.maximum(np.abs(u), np.abs(v)) <= r)
    elif k == 8:  # vertical bars
        mask = (np.abs((u + r) % (0.7 * r) - 0.35 * r) <= 0.6 * w) & (np.maximum(np.abs(u), np.abs(v)) <= r)
    else:         # ellipse
        mask = (u / r) ** 2 + (v / (0.5 * r)) ** 2 <= 1
    return mask.astype(np.float32)


def make_shapes(n: int, size: int = 28, noise: float = 0.25, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n images of random shapes.

    Returns a pair (X, T), with X a float32 array of shape (n, size * size) with values in [0, 1],
    and T an int64 array of shape (n,) with class indices into SHAPE_CLASSES.
    """
    rng = np.random.default_rng(seed)
    T = rng.integers(0, len(SHAPE_CLASSES), n)

    def uniform(low, high):
        return rng.uniform(low, high, (n, 1, 1)).astype(np.float32)

    margin = 10 * size / 28
    cx = uniform(margin, size - margin)     # center
    cy = uniform(margin, size - margin)
    r = uniform(0.15 * size, 0.29 * size)   # radius
    w = uniform(0.035 * size, 0.08 * size)  # stroke width
    a = uniform(-0.5, 0.5)                  # rotation in radians

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    X = np.empty((n, size, size), dtype=np.float32)
    for k in range(len(SHAPE_CLASSES)):
        I = np.flatnonzero(T == k)
        X[I] = _draw_shape(k, xx[None] - cx[I], yy[None] - cy[I], r[I], w[I], a[I])
    X *= uniform(0.5, 1.0)                                       # brightness
    X += noise * rng.standard_normal(X.shape, dtype=np.float32)  # pixel noise
    X = np.clip(X, 0.0, 1.0).reshape(n, size * size)
    return X, T.astype(np.int64)


def make_shapes_dataset(num_train: int = 50000, num_test: int = 10000, size: int = 28, noise: float = 0.25,
                        seed: int = 0) -> Dict[str, np.ndarray]:
    """Generate a training and a test set of random shapes.

    Returns a dictionary with the numpy arrays Xtrain, Ttrain, Xtest and Ttest, in the same format as
    the .npz files that are used by create_npz_dataloaders.
    """
    train_seed, test_seed = np.random.SeedSequence(seed).spawn(2)
    Xtrain, Ttrain = make_shapes(num_train, size, noise, train_seed)
    Xtest, Ttest = make_shapes(num_test, size, noise, test_seed)
    return {'Xtrain': Xtrain, 'Ttrain': Ttrain, 'Xtest': Xtest, 'Ttest': Ttest}


def create_shapes_dataloaders(batch_size: int = 100, num_train: int = 50000, num_test: int = 10000,
                              size: int = 28, noise: float = 0.25, seed: int = 0):
    """Generate a dataset of random shapes, and return a data loader for the training and the test set."""
    from nerva_jax.datasets import create_array_dataloaders
    data = make_shapes_dataset(num_train, num_test, size, noise, seed)
    return create_array_dataloaders(data['Xtrain'], data['Ttrain'], data['Xtest'], data['Ttest'], batch_size,
                                    num_classes=len(SHAPE_CLASSES))


def main():
    parser = argparse.ArgumentParser(description='Generate the synthetic shapes dataset and save it to a .npz file.')
    parser.add_argument('--output', type=str, default='shapes.npz', help='The output file (default: shapes.npz)')
    parser.add_argument('--num-train', type=int, default=50000, help='The number of training examples (default: 50000)')
    parser.add_argument('--num-test', type=int, default=10000, help='The number of test examples (default: 10000)')
    parser.add_argument('--size', type=int, default=28, help='The width and height of the images (default: 28)')
    parser.add_argument('--noise', type=float, default=0.25, help='The standard deviation of the pixel noise (default: 0.25)')
    parser.add_argument('--seed', type=int, default=0, help='The random seed (default: 0)')
    args = parser.parse_args()

    data = make_shapes_dataset(args.num_train, args.num_test, args.size, args.noise, args.seed)
    np.savez_compressed(args.output, **data)
    print(f"Saved the shapes dataset to '{args.output}'")


if __name__ == '__main__':
    main()
