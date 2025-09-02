# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Training helpers for the MLP, including a basic SGD loop and CLI glue."""

import random
from typing import List
import jax.numpy as jnp

from nerva_jax.datasets import DataLoader, create_npz_dataloaders, to_one_hot
from nerva_jax.learning_rate import LearningRateScheduler, parse_learning_rate
from nerva_jax.loss_functions import LossFunction, parse_loss_function
from nerva_jax.matrix_operations import Matrix
from nerva_jax.multilayer_perceptron import MultilayerPerceptron, parse_multilayer_perceptron
from nerva_jax.utilities import StopWatch, pp, set_jax_options


class SGDOptions(object):
    debug=False


def print_epoch(epoch, lr, loss, train_accuracy, test_accuracy, elapsed):
    """Print formatted training statistics for one epoch."""
    print(f'epoch {epoch:3}  '
          f'lr: {lr:.8f}  '
          f'loss: {loss:.8f}  '
          f'train accuracy: {train_accuracy:.8f}  '
          f'test accuracy: {test_accuracy:.8f}  '
          f'time: {elapsed:.8f}s'
         )


def compute_accuracy(M: MultilayerPerceptron, data_loader: DataLoader):
    """Compute mean classification accuracy for a model over a data loader."""
    N = len(data_loader.dataset)  # N is the number of examples
    total_correct = 0
    for X, T in data_loader:
        Y = M.feedforward(X)
        predicted = Y.argmax(axis=1)  # the predicted classes for the batch
        targets = T.argmax(axis=1)    # the expected classes
        total_correct += (predicted == targets).sum().item()
    return total_correct / N


def compute_loss(M: MultilayerPerceptron, data_loader: DataLoader, loss: LossFunction):
    """Compute mean loss for a model over a data loader using the given loss."""
    N = len(data_loader.dataset)  # N is the number of examples
    total_loss = 0.0
    for X, T in data_loader:
        Y = M.feedforward(X)
        total_loss += loss(Y, T)
    return total_loss / N


def compute_statistics(M, lr, loss, train_loader, test_loader, epoch, elapsed_seconds=0.0, print_statistics=True):
    """Compute and optionally print loss and accuracy statistics."""
    if print_statistics:
        train_loss = compute_loss(M, train_loader, loss)
        train_accuracy = compute_accuracy(M, train_loader)
        test_accuracy = compute_accuracy(M, test_loader)
        print_epoch(epoch, lr, train_loss, train_accuracy, test_accuracy, elapsed_seconds)
    else:
        print(f'epoch {epoch:3}')


def print_batch_debug_info(epoch: int, batch_idx: int,
                           M: MultilayerPerceptron,
                           X: Matrix, Y: Matrix, DY: Matrix):
    """Print detailed debug information for a training batch."""
    print(f'epoch: {epoch} batch: {batch_idx}')
    M.info()
    pp("X", X)
    pp("Y", Y)
    pp("DY", DY)


# tag::sgd[]
def stochastic_gradient_descent(M: MultilayerPerceptron,
                                epochs: int,
                                loss: LossFunction,
                                learning_rate: LearningRateScheduler,
                                train_loader: DataLoader,
                                test_loader: DataLoader
                                ):
# end::sgd[]
    """
    Run a simple stochastic gradient descent (SGD) training loop using PyTorch data loaders.

    Args:
        M (MultilayerPerceptron): The neural network model to train.
        epochs (int): Number of training epochs.
        loss (LossFunction): The loss function instance (must provide `gradient` method).
        learning_rate (LearningRateScheduler): Scheduler returning the learning rate per epoch.
        train_loader (DataLoader): DataLoader that yields mini-batches `(X, T)` for training.

            - `X`: input batch of shape (batch_size, input_dim).
            - `T`: batch of target labels, either class indices (batch_size,) or one-hot
              encoded (batch_size, num_classes).
        test_loader (DataLoader): DataLoader that yields test batches `(X, T)` for evaluation.

    Notes:
        - The learning rate is updated once per epoch using the scheduler.
        - Gradients are normalized by batch size before backpropagation.
        - Debugging output is controlled by `SGDOptions.debug`. When enabled,
          per-batch information is printed via `print_batch_debug_info`.

    Side Effects:
        - Updates model parameters in-place via `M.optimize(lr)`.
        - Prints statistics and training time to standard output.
    """
# tag::sgd[]
    lr = learning_rate(0)
    compute_statistics(M, lr, loss, train_loader, test_loader, epoch=0)
    training_time = 0.0

    for epoch in range(epochs):
        timer = StopWatch()
        lr = learning_rate(epoch)  # update the learning at the start of each epoch

        for k, (X, T) in enumerate(train_loader):
            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / X.shape[0]

            if SGDOptions.debug:
                print_batch_debug_info(epoch, k, M, X, Y, DY)

            M.backpropagate(Y, DY)
            M.optimize(lr)

        seconds = timer.seconds()
        training_time += seconds
        compute_statistics(M, lr, loss, train_loader, test_loader, epoch=epoch + 1, elapsed_seconds=seconds)

    print(f'Total training time for the {epochs} epochs: {training_time:.8f}s\n')
# end::sgd[]


# tag::sgd_plain[]
def stochastic_gradient_descent_plain(M: MultilayerPerceptron,
                                      Xtrain: Matrix,
                                      Ttrain: Matrix,
                                      loss: LossFunction,
                                      learning_rate: LearningRateScheduler,
                                      epochs: int,
                                      batch_size: int,
                                      shuffle: bool
                                     ):
# end::sgd_plain[]
    """
    Perform plain stochastic gradient descent training for a multilayer perceptron
    using raw tensors in row layout (samples are rows).

    Args:
        M (MultilayerPerceptron): The neural network model to train.
        Xtrain: Training input data of shape (N, input_dim),
            where N is the number of training examples.
        Ttrain: Training labels. Either:
            - class indices of shape (N,) or (N, 1), or
            - one-hot encoded labels of shape (N, num_classes).
        loss (LossFunction): The loss function instance (with `gradient` method).
        learning_rate (LearningRateScheduler): Scheduler returning the learning rate per epoch.
        epochs (int): Number of training epochs.
        batch_size (int): Number of examples per mini-batch.
        shuffle (bool): Whether to shuffle training examples each epoch.

    Notes:
        - The learning rate is updated once per epoch using the scheduler.
        - Gradients are normalized by batch size before backpropagation.
        - Debugging output is controlled by `SGDOptions.debug`. When enabled,
          per-batch information is printed via `print_batch_debug_info`.
        - If `Ttrain` contains class indices, they will be converted to one-hot encoding.

    Side Effects:
        - Updates model parameters in-place via `M.optimize(lr)`.
        - Prints statistics and training time to standard output.
    """
# tag::sgd_plain[]
    N = Xtrain.shape[0]  # number of examples (row layout)
    I = list(range(N))
    K = N // batch_size  # number of full batches
    num_classes = M.layers[-1].output_size()

    for epoch in range(epochs):
        if shuffle:
            random.shuffle(I)
        lr = learning_rate(epoch)  # update learning rate each epoch

        for k in range(K):
            batch = I[k * batch_size: (k + 1) * batch_size]
            X = Xtrain[batch, :]   # shape (batch_size, input_dim)

            # Convert labels to one-hot if needed
            if Ttrain.ndim == 2 and Ttrain.shape[1] > 1:
                # already one-hot encoded
                T = Ttrain[batch, :]
            else:
                T = to_one_hot(Ttrain[batch], num_classes)

            Y = M.feedforward(X)
            DY = loss.gradient(Y, T) / X.shape[0]

            if SGDOptions.debug:
                print_batch_debug_info(epoch, k, M, X, Y, DY)

            M.backpropagate(Y, DY)
            M.optimize(lr)
# end::sgd_plain[]


def train(layer_specifications: List[str],
          linear_layer_sizes: List[int],
          linear_layer_optimizers: List[str],
          linear_layer_weight_initializers: List[str],
          batch_size: int,
          epochs: int,
          loss: str,
          learning_rate: str,
          weights_and_bias_file: str,
          dataset_file: str,
          debug: bool
         ):
    """High-level training convenience that wires parsing, data and SGD."""

    SGDOptions.debug = debug
    set_jax_options()
    loss = parse_loss_function(loss)
    learning_rate = parse_learning_rate(learning_rate)
    M = parse_multilayer_perceptron(layer_specifications, linear_layer_sizes, linear_layer_optimizers, linear_layer_weight_initializers)
    if weights_and_bias_file:
        M.load_weights_and_bias(weights_and_bias_file)
    train_loader, test_loader = create_npz_dataloaders(dataset_file, batch_size=batch_size)
    stochastic_gradient_descent(M, epochs, loss, learning_rate, train_loader, test_loader)
