# MNIST and CIFAR-10 examples

This folder contains examples that train MLPs on the [MNIST](https://en.wikipedia.org/wiki/MNIST_database)
and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. They are optional: the examples in the
parent folder and the tests do not need these datasets. They use a synthetic dataset that is generated on the
fly (see `nerva_jax.shapes_dataset`).

## Installation

The examples only need `jax` and the `nerva-jax` package:

```bash
pip install nerva-jax       # or, from the root of the repository: pip install -e .
```

The datasets have to be downloaded once (about 11 MB for MNIST and 170 MB for CIFAR-10). The script
`prepare_data.py` downloads them, verifies their checksums, and stores a flattened and normalized version in
the `data` subfolder:

```bash
cd examples/mnist_cifar10
python prepare_data.py --dataset=mnist --download --data-dir=data
python prepare_data.py --dataset=cifar10 --download --data-dir=data
```

This creates `data/mnist-flattened.npz` and `data/cifar10-flattened.npz`. They contain the arrays `Xtrain`,
`Ttrain`, `Xtest` and `Ttest`, where `Xtrain` and `Xtest` have dtype float32 with values in [0, 1] and
`Ttrain` and `Ttest` contain the class labels. Existing files are not downloaded or processed again.

## Examples

| Example | Description |
|---------|-------------|
| `train_mnist.py` | Trains an MLP on MNIST with an explicit mini-batch SGD loop |
| `train_mnist_cli.sh` | Trains an MLP on MNIST with the `mlp` command line tool (`tools/mlp.py`) |
| `train_cifar10_cli.sh` | Trains an MLP on CIFAR-10 with the `mlp` command line tool |

The examples can be run from any directory, e.g.

```bash
python examples/mnist_cifar10/train_mnist.py
bash examples/mnist_cifar10/train_mnist_cli.sh
bash examples/mnist_cifar10/train_cifar10_cli.sh
```

The shell scripts use the local source code in `src`; `train_mnist.py` uses the installed package.
