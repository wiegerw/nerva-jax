# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""In-memory data loader helpers and one-hot conversions.

The DataLoader defined here mirrors a small subset of the PyTorch
DataLoader API but operates on in-memory tensors loaded from .npz files.
"""

from pathlib import Path
from typing import Union, Tuple

import jax.numpy as jnp
from nerva_jax.matrix_operations import Matrix
from nerva_jax.utilities import load_dict_from_npz


def to_one_hot(x: Matrix, num_classes: int):
    """Convert class index tensor to one-hot matrix with num_classes columns."""
    one_hot = jnp.zeros((len(x), num_classes), dtype=float)
    one_hot = one_hot.at[jnp.arange(len(x)), x].set(1)
    return jnp.array(one_hot)


def from_one_hot(one_hot: Matrix) -> Matrix:
    """Convert one-hot encoded rows to class index tensor."""
    return jnp.argmax(one_hot, axis=1)


class DataLoader(object):
    """A minimal in-memory data loader with an interface similar to torch.utils.data.DataLoader.

    Notes / Warning:

    - When `Tdata` contains class indices (shape (N,) or (N,1)), this loader will one-hot encode
      the labels. If `num_classes` is not provided, it will be inferred as `max(Tdata) + 1`.
    - On small datasets or subsets where some classes are absent, this inference can underestimate
      the true number of classes and produce one-hot targets with too few columns. This may cause
      dimension mismatches with the model output during training/evaluation.
    - To avoid this, pass `num_classes` explicitly whenever you know the total number of classes.
    """

    def __init__(self, Xdata: Matrix, Tdata: Matrix, batch_size: int, num_classes=0):
        """Iterate batches over row-major tensors; one-hot encode targets if needed.

        If Tdata is a vector of class indices and num_classes > 0 (or can be
        inferred), batches yield (X, one_hot(T)). Otherwise, targets are returned as-is.
        """
        self.Xdata = Xdata
        self.Tdata = Tdata
        self.batch_size = batch_size
        self.dataset = Xdata
        self.num_classes = int(Tdata.max() + 1) if num_classes == 0 and len(Tdata.shape) == 1 else num_classes

    def __iter__(self):
        N = self.Xdata.shape[0]  # total number of examples
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch = slice(start, end)
            Xbatch = self.Xdata[batch]
            Tbatch = self.Tdata[batch]
            if self.num_classes:
                Tbatch = to_one_hot(Tbatch, self.num_classes)
            yield Xbatch, Tbatch

    def __len__(self):
        """Number of batches (including incomplete last batch)."""
        N = self.Xdata.shape[0]
        return (N + self.batch_size - 1) // self.batch_size  # ceiling division

    @property
    def dataset_size(self):
        """Total number of examples."""
        return int(self.Xdata.shape[0])


def max_(X: Matrix) -> Union[int, float]:
    """Return the maximum element of X as a Python scalar."""
    return jnp.max(X).item()


def infer_num_classes(Ttrain: Matrix, Ttest: Matrix) -> int:
    """Infer total number of classes from targets.

    - If either Ttrain or Ttest is one-hot encoded (2D with width > 1), use that width.
    - Otherwise assume class indices and return max over both + 1.
    """
    if len(Ttrain.shape) == 2 and Ttrain.shape[1] > 1:
        return int(Ttrain.shape[1])
    if len(Ttest.shape) == 2 and Ttest.shape[1] > 1:
        return int(Ttest.shape[1])

    max_train = max_(Ttrain)
    max_test = max_(Ttest)

    return int(max(max_train, max_test) + 1)


def create_npz_dataloaders(filename: str, batch_size: int=True) -> Tuple[DataLoader, DataLoader]:
    """Creates a data loader from a file containing a dictionary with Xtrain, Ttrain, Xtest and Ttest tensors."""
    path = Path(filename)
    print(f'Loading dataset from file {path}')
    if not path.exists():
        raise RuntimeError(f"Could not load file '{path}'")

    data = load_dict_from_npz(filename)
    Xtrain, Ttrain, Xtest, Ttest = data['Xtrain'], data['Ttrain'], data['Xtest'], data['Ttest']

    # Determine number of classes robustly to avoid underestimating when some classes are absent
    num_classes = infer_num_classes(Ttrain, Ttest)

    train_loader = DataLoader(jnp.array(Xtrain), jnp.array(Ttrain), batch_size, num_classes=num_classes)
    test_loader = DataLoader(jnp.array(Xtest), jnp.array(Ttest), batch_size, num_classes=num_classes)
    return train_loader, test_loader
