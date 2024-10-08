# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import numpy as np


def to_one_hot(x: jnp.ndarray, n_classes: int):
    """
    Converts a tensor of class indices to a one-hot encoded tensor.

    Args:
        x (torch.LongTensor): Tensor of class indices.
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: One-hot encoded tensor of shape (len(x), num_classes).
    """
    one_hot = np.zeros((len(x), n_classes), dtype=float)
    one_hot[np.arange(len(x)), x] = 1
    return jnp.array(one_hot)


def from_one_hot(one_hot: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a one-hot encoded tensor back to a tensor of class indices.

    Args:
        one_hot (jnp.ndarray): One-hot encoded tensor of shape (N, num_classes).

    Returns:
        jnp.ndarray: Tensor of class indices of shape (N,).
    """
    return jnp.argmax(one_hot, axis=1)


class MemoryDataLoader(object):
    """
    A data loader with an interface similar to torch.utils.data.DataLoader.
    """

    def __init__(self, Xdata: jnp.ndarray, Tdata: jnp.ndarray, batch_size: int=True, num_classes=0):
        """
        :param Xdata: a dataset with row layout
        :param Tdata: the expected targets. In case of a classification task the targets may be specified as a vector
                      of integers that denote the expected classes. In such a case the targets will be expanded on the
                      fly using one hot encoding.
        :param batch_size: the batch size
        :param num_classes: the number of classes in case of a classification problem, 0 otherwise
        """
        self.Xdata = Xdata
        self.Tdata = Tdata
        self.batch_size = batch_size
        self.dataset = Xdata
        self.num_classes = int(Tdata.max() + 1) if num_classes == 0 and len(Tdata.shape) == 1 else num_classes

    def __iter__(self):
        N = self.Xdata.shape[0]  # N is the number of examples
        K = N // self.batch_size  # K is the number of batches
        for k in range(K):
            batch = jnp.array(range(k * self.batch_size, (k + 1) * self.batch_size))
            Xbatch = self.Xdata[batch]
            Tbatch = self.Tdata[batch]
            yield Xbatch, to_one_hot(Tbatch, self.num_classes) if self.num_classes else Tbatch
    def __len__(self):
        """
        Returns the number of batches
        """
        return self.Xdata.shape[0] // self.batch_size


DataLoader = MemoryDataLoader


def create_npz_dataloaders(filename: str, batch_size: int=True) -> Tuple[MemoryDataLoader, MemoryDataLoader]:
    """
    Creates a data loader from a file containing a dictionary with Xtrain, Ttrain, Xtest and Ttest tensors
    :param filename: a file in NumPy .npz format
    :param batch_size: the batch size of the data loader
    :return: a tuple of data loaders
    """
    path = Path(filename)
    print(f'Loading dataset from file {path}')
    if not path.exists():
        raise RuntimeError(f"Could not load file '{path}'")

    data = dict(jnp.load(filename, allow_pickle=True))
    Xtrain, Ttrain, Xtest, Ttest = data['Xtrain'], data['Ttrain'], data['Xtest'], data['Ttest']
    train_loader = MemoryDataLoader(jnp.array(Xtrain), jnp.array(Ttrain), batch_size)
    test_loader = MemoryDataLoader(jnp.array(Xtest), jnp.array(Ttest), batch_size)
    return train_loader, test_loader
