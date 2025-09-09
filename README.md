# nerva-jax

[![PyPI](https://img.shields.io/pypi/v/nerva-jax.svg)](https://pypi.org/project/nerva-jax/)
[![License: BSL-1.0](https://img.shields.io/badge/license-BSL%201.0-blue.svg)](https://opensource.org/licenses/BSL-1.0)

**`nerva-jax`** is a minimal, transparent implementation of multilayer perceptrons using **JAX** tensors.  
It is part of the [Nerva](https://github.com/wiegerw/nerva) project — a suite of Python and C++ libraries that provide well-specified, inspectable implementations of neural networks.

➡️ **All equations in this repository are written in *batch (minibatch) matrix form***, meaning feedforward, backpropagation, and loss functions operate on minibatches of inputs rather than single examples.

## 🗺️ Overview

The `nerva` libraries aim to demystify neural networks by:
- Providing precise mathematical specifications.
- Implementing core concepts like backpropagation from scratch.
- Avoiding automatic differentiation to foster understanding.

Currently supported: **Multilayer Perceptrons (MLPs)**.  
Future extensions to convolutional or recurrent networks are possible.

---

## ❓ Why Use `nerva`

If you're learning or teaching how neural networks work, most modern frameworks (e.g., PyTorch, TensorFlow) can be too opaque. `nerva` is different:

- Every function has a clear mathematical interpretation.
- Gradient computations are written by hand — no autograd.
- Includes symbolic validation to ensure correctness.
- Modular and backend-agnostic: choose between JAX, NumPy, PyTorch, or TensorFlow.
- Used as a reference implementation for research and education.
- Modularity: the core operations rely on a small set of primitive [matrix operations](https://wiegerw.github.io/nerva-jax/nerva-jax.html#_matrix_operations), making the logic easy to inspect, test, and validate.

---

## 📦 Available Python Packages

Each backend has a dedicated PyPI package and GitHub repository:

| Package             | Backend     | PyPI                                               | GitHub                                                  |
|---------------------|-------------|----------------------------------------------------|----------------------------------------------------------|
| `nerva-jax`         | JAX         | [nerva-jax](https://pypi.org/project/nerva-jax/)           | [repo](https://github.com/wiegerw/nerva-jax)            |
| `nerva-numpy`       | NumPy       | [nerva-numpy](https://pypi.org/project/nerva-numpy/)       | [repo](https://github.com/wiegerw/nerva-numpy)          |
| `nerva-tensorflow`  | TensorFlow  | [nerva-tensorflow](https://pypi.org/project/nerva-tensorflow/) | [repo](https://github.com/wiegerw/nerva-tensorflow)     |
| `nerva-torch`       | PyTorch     | [nerva-torch](https://pypi.org/project/nerva-torch/)       | [repo](https://github.com/wiegerw/nerva-torch)          |
| `nerva-sympy`       | SymPy       | [nerva-sympy](https://pypi.org/project/nerva-sympy/)       | [repo](https://github.com/wiegerw/nerva-sympy)          |

> 📝 `nerva-sympy` is intended for validation and testing — it depends on the other four.

See the [nerva meta-repo](https://github.com/wiegerw/nerva) for an overview of all Python and C++ variants.

---

## 🚀 Quick Start

### Installation

The library can be installed in two ways: from the source repository or from the Python Package Index (PyPI).

```bash
# Install from the local repository
pip install .
```

```bash
# Install directly from PyPI
pip install nerva-jax
```

### Example: Define and Train an MLP

```python
# Create a new MLP model
M = MultilayerPerceptron()
M.layers = [
    ActivationLayer(784, 1024, ReLUActivation()),
    ActivationLayer(1024, 512, ReLUActivation()),
    LinearLayer(512, 10)
]
for layer in M.layers:
    layer.set_optimizer("Momentum(0.9)")
    layer.set_weights("XavierNormal")

loss = StableSoftmaxCrossEntropyLossFunction()
learning_rate = ConstantScheduler(0.01)
epochs = 10

# Load data
train_loader, test_loader = create_npz_dataloaders("../data/mnist-flattened.npz", batch_size=100)

# Train the network
stochastic_gradient_descent(M, epochs, loss, learning_rate, train_loader, test_loader)
```

## 🧱 Architecture

Each major concept is implemented through clear interface classes. Implementations are modular and easy to replace:

| Concept               | Interface Class        | Example Implementations                         |
|------------------------|------------------------|--------------------------------------------------|
| Layer                 | `Layer`                | `ActivationLayer`, `LinearLayer`                |
| Activation Function   | `ActivationFunction`   | `ReLUActivation`, `SigmoidActivation`           |
| Loss Function         | `LossFunction`         | `SoftmaxCrossEntropyLossFunction`               |
| Optimizer             | `Optimizer`            | `GradientDescentOptimizer`, `MomentumOptimizer` |
| Learning Rate Schedule| `LearningRateScheduler`| `ConstantScheduler`, `ExponentialScheduler`     |

---

## 🛠 Features

- Feedforward and backpropagation logic match documented equations exactly.
- Formulas use batch matrix form, enabling efficient computation over minibatches.
- Customizable optimizers per parameter group using a composite pattern.
- Symbolic gradient validation using [nerva-sympy](https://github.com/wiegerw/nerva-sympy).
- Lightweight command-line interface for experiments.

---

## 📚 Documentation

The full documentation is hosted on GitHub Pages:

- [📖 Documentation Landing Page](https://wiegerw.github.io/nerva-jax/)

From there you can access:
- [API Reference (Sphinx)](https://wiegerw.github.io/nerva-jax/sphinx/)
- [nerva-jax Manual](https://wiegerw.github.io/nerva-jax/nerva-jax.html)
- [Mathematical Specifications (PDF)](https://wiegerw.github.io/nerva-rowwise/pdf/nerva-library-specifications.pdf)


Relevant papers:

1. [**Nerva: a Truly Sparse Implementation of Neural Networks**](https://arxiv.org/abs/2407.17437)
2. _Batch Matrix-form Equations and Implementation of Multilayer Perceptrons_ (🔗 TODO)

---

## 🧪 Training Loop Internals

A mini-batch gradient descent loop with forward, backward, and optimizer steps can be implemented in just a few lines of code:

```python
def stochastic_gradient_descent(M: MultilayerPerceptron,
                                epochs: int,
                                loss: LossFunction,
                                learning_rate: LearningRateScheduler,
                                train_loader: DataLoader):

    for epoch in range(epochs):
        lr = learning_rate(epoch)

        # Iterate over mini-batches X with target T
        for (X, T) in train_loader:
            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / Y.shape[0]
            M.backpropagate(Y, DY)
            M.optimize(lr)
```

---

## ✅ Symbolic Validation (Softmax Layer Example)

We validate the manually written backpropagation code using symbolic differentiation via [SymPy](https://www.sympy.org/).

This example validates the gradient of the **softmax layer**. It also illustrates how the gradients `DZ`, `DW`, `Db` and `DX` of the intermediate variable `Z`, the weights `W`, bias `b` and input `X` are calculated from the output `Y` and its gradient `DY`.

```python
# Backpropagation gradients
DZ = hadamard(Y, DY - row_repeat(diag(Y.T * DY).T, K))
DW = DZ * X.T
Db = rows_sum(DZ)
DX = W.T * DZ

# Symbolic comparison
DW1 = gradient(loss(Y), w)
assert equal_matrices(DW, DW1)
```

## 🔢 Implementation via Matrix Operations

The validated backpropagation formulae are implemented directly using the library's core set of primitive matrix operations. This approach provides a significant advantage in clarity and maintainability by expressing all computations from loss functions and activation layers to gradient calculations through a single, global vocabulary of operations.

This stands in contrast to implementations that use hundreds of lines of scattered, special-case logic for the same mathematical result. By reducing complex formulae to a concise sequence of well-defined primitives, the implementation becomes both more readable and far easier to verify and debug.

For a complete reference of all available operations, see the [Table of Matrix Operations](https://wiegerw.github.io/nerva-jax/nerva-jax.html#_matrix_operations).

---

## 📜 License

Distributed under the [Boost Software License 1.0](http://www.boost.org/LICENSE_1_0.txt).  
[License file](https://github.com/wiegerw/nerva-jax/blob/main/LICENSE)

---

## 🙋 Contributing

Bug reports and contributions are welcome via the [GitHub issue tracker](https://github.com/wiegerw/nerva-jax/issues).

