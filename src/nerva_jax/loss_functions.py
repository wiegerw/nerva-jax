# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Analytic loss functions and their gradients used during training.

Functions are provided in vector (lowercase) and matrix (uppercase) forms.
Concrete LossFunction classes wrap these for use in the training loop.
"""

import jax.numpy as jnp
from nerva_jax.activation_functions import Sigmoid
from nerva_jax.matrix_operations import column_repeat, dot, elements_sum, hadamard, log, log_sigmoid, reciprocal, rows_sum, Matrix
from nerva_jax.softmax_functions import log_softmax, softmax, stable_log_softmax, stable_softmax


# Naming conventions:
# - lowercase functions operate on vectors (y and t)
# - uppercase functions operate on matrices (Y and T)


def squared_error_loss(y, t):
    """Squared error loss for vectors: ||y - t||²."""
    return dot(y - t, y - t)


def squared_error_loss_gradient(y, t):
    """Gradient of squared error loss for vectors."""
    return 2 * (y - t)


def Squared_error_loss(Y, T):
    """Squared error loss for matrices: sum of ||Y - T||²."""
    return elements_sum(hadamard(Y - T, Y - T))


def Squared_error_loss_gradient(Y, T):
    """Gradient of squared error loss for matrices."""
    return 2 * (Y - T)


def cross_entropy_loss(y, t):
    """Cross entropy loss for vectors: -t^T log(y)."""
    return -dot(t, log(y))


def cross_entropy_loss_gradient(y, t):
    """Gradient of cross entropy loss for vectors."""
    return -hadamard(t, reciprocal(y))


def Cross_entropy_loss(Y, T):
    """Cross entropy loss for matrices: -sum(T ⊙ log(Y))."""
    return -elements_sum(hadamard(T, log(Y)))


def Cross_entropy_loss_gradient(Y, T):
    """Gradient of cross entropy loss for matrices."""
    return -hadamard(T, reciprocal(Y))


def softmax_cross_entropy_loss(y, t):
    """Softmax cross entropy loss for vectors."""
    return -dot(t, log_softmax(y))


def softmax_cross_entropy_loss_gradient(y, t):
    """Gradient of softmax cross entropy loss for vectors."""
    return elements_sum(t) * softmax(y) - t


def softmax_cross_entropy_loss_gradient_one_hot(y, t):
    """Gradient of softmax cross entropy for one-hot targets."""
    return softmax(y) - t


def Softmax_cross_entropy_loss(Y, T):
    """Softmax cross entropy loss for matrices."""
    return -elements_sum(hadamard(T, log_softmax(Y)))


def Softmax_cross_entropy_loss_gradient(Y, T):
    """Gradient of softmax cross entropy loss for matrices."""
    N, K = Y.shape
    return hadamard(softmax(Y), column_repeat(rows_sum(T), K)) - T


def Softmax_cross_entropy_loss_gradient_one_hot(Y, T):
    """Gradient of softmax cross entropy for one-hot targets (matrices)."""
    return softmax(Y) - T


def stable_softmax_cross_entropy_loss(y, t):
    """Stable softmax cross entropy loss for vectors."""
    return -dot(t, stable_log_softmax(y))


def stable_softmax_cross_entropy_loss_gradient(y, t):
    """Gradient of stable softmax cross entropy loss for vectors."""
    return stable_softmax(y) * elements_sum(t) - t


def stable_softmax_cross_entropy_loss_gradient_one_hot(y, t):
    """Gradient of stable softmax cross entropy for one-hot targets."""
    return stable_softmax(y) - t


def Stable_softmax_cross_entropy_loss(Y, T):
    """Stable softmax cross entropy loss for matrices."""
    return -elements_sum(hadamard(T, stable_log_softmax(Y)))


def Stable_softmax_cross_entropy_loss_gradient(Y, T):
    """Gradient of stable softmax cross entropy loss for matrices."""
    N, K = Y.shape
    return hadamard(stable_softmax(Y), column_repeat(rows_sum(T), K)) - T


def Stable_softmax_cross_entropy_loss_gradient_one_hot(Y, T):
    """Gradient of stable softmax cross entropy for one-hot targets (matrices)."""
    return stable_softmax(Y) - T


def logistic_cross_entropy_loss(y, t):
    """Logistic cross entropy loss for vectors."""
    return -dot(t, log_sigmoid(y))


def logistic_cross_entropy_loss_gradient(y, t):
    """Gradient of logistic cross entropy loss for vectors."""
    return hadamard(t, Sigmoid(y)) - t


def Logistic_cross_entropy_loss(Y, T):
    """Logistic cross entropy loss for matrices."""
    return -elements_sum(hadamard(T, log_sigmoid(Y)))


def Logistic_cross_entropy_loss_gradient(Y, T):
    """Gradient of logistic cross entropy loss for matrices."""
    return hadamard(T, Sigmoid(Y)) - T


def negative_log_likelihood_loss(y, t):
    """Negative log likelihood loss for vectors."""
    return -log(dot(y, t))


def negative_log_likelihood_loss_gradient(y, t):
    """Gradient of negative log likelihood loss for vectors."""
    return -reciprocal(dot(y, t)) * t


def Negative_log_likelihood_loss(Y, T):
    """Negative log likelihood loss for matrices."""
    return -elements_sum(log(rows_sum(hadamard(Y, T))))


def Negative_log_likelihood_loss_gradient(Y, T):
    """Gradient of negative log likelihood loss for matrices."""
    N, K = Y.shape
    return -hadamard(column_repeat(reciprocal(rows_sum(hadamard(Y, T))), K), T)


class LossFunction(object):
    """Interface for loss functions with value and gradient on batch matrices."""
    def __call__(self, Y: Matrix, T: Matrix):
        raise NotImplementedError

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        raise NotImplementedError


class SquaredErrorLossFunction(LossFunction):
    """Squared error loss function for regression tasks."""
    def __call__(self, Y: Matrix, T: Matrix):
        return Squared_error_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Squared_error_loss_gradient(Y, T)


class CrossEntropyLossFunction(LossFunction):
    """Cross entropy loss function for classification with probabilities."""
    def __call__(self, Y: Matrix, T: Matrix):
        return Cross_entropy_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Cross_entropy_loss_gradient(Y, T)


class SoftmaxCrossEntropyLossFunction(LossFunction):
    """Softmax cross entropy loss for classification with logits."""
    def __call__(self, Y: Matrix, T: Matrix):
        return Softmax_cross_entropy_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Softmax_cross_entropy_loss_gradient(Y, T)


class StableSoftmaxCrossEntropyLossFunction(LossFunction):
    """Numerically stable softmax cross entropy loss for classification."""
    def __call__(self, Y: Matrix, T: Matrix):
        return Stable_softmax_cross_entropy_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Stable_softmax_cross_entropy_loss_gradient(Y, T)


class LogisticCrossEntropyLossFunction(LossFunction):
    """Logistic cross entropy loss for binary classification."""
    def __call__(self, Y: Matrix, T: Matrix):
        return Logistic_cross_entropy_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Logistic_cross_entropy_loss_gradient(Y, T)


class NegativeLogLikelihoodLossFunction(LossFunction):
    """Negative log likelihood loss for probabilistic outputs."""
    def __call__(self, Y: Matrix, T: Matrix):
        return Negative_log_likelihood_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Negative_log_likelihood_loss_gradient(Y, T)


def parse_loss_function(text: str) -> LossFunction:
    """Parse a loss function name into a LossFunction instance.

    Supported names: SquaredError, CrossEntropy, SoftmaxCrossEntropy,
    LogisticCrossEntropy, NegativeLogLikelihood.
    """
    if text == "SquaredError":
        return SquaredErrorLossFunction()
    elif text == "CrossEntropy":
        return CrossEntropyLossFunction()
    elif text == "SoftmaxCrossEntropy":
        return StableSoftmaxCrossEntropyLossFunction()
    elif text == "LogisticCrossEntropy":
        return LogisticCrossEntropyLossFunction()
    elif text == "NegativeLogLikelihood":
        return NegativeLogLikelihoodLossFunction()
    else:
        raise RuntimeError(f"unknown loss function '{text}'")
