# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import jax.numpy as jnp
from nerva_jax.activation_functions import Sigmoid
from nerva_jax.matrix_operations import column_repeat, dot, elements_sum, hadamard, log, log_sigmoid, reciprocal, \
    rows_sum
from nerva_jax.softmax_functions import log_softmax, softmax, stable_log_softmax, stable_softmax


# Naming conventions:
# - lowercase functions operate on vectors (y and t)
# - uppercase functions operate on matrices (Y and T)


def squared_error_loss(y, t):
    return dot(y - t, y - t)


def squared_error_loss_gradient(y, t):
    return 2 * (y - t)


def Squared_error_loss(Y, T):
    return elements_sum(hadamard(Y - T, Y - T))


def Squared_error_loss_gradient(Y, T):
    return 2 * (Y - T)


def mean_squared_error_loss(y, t):
    N, K = y.shape
    return squared_error_loss(y, t) / K


def mean_squared_error_loss_gradient(y, t):
    N, K = y.shape
    return squared_error_loss_gradient(y, t) / K


def Mean_squared_error_loss(Y, T):
    N, K = Y.shape
    return Squared_error_loss(Y, T) / (K * N)


def Mean_squared_error_loss_gradient(Y, T):
    N, K = Y.shape
    return Squared_error_loss_gradient(Y, T) / (K * N)


def cross_entropy_loss(y, t):
    return -dot(t, log(y))


def cross_entropy_loss_gradient(y, t):
    return -hadamard(t, reciprocal(y))


def Cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, log(Y)))


def Cross_entropy_loss_gradient(Y, T):
    return -hadamard(T, reciprocal(Y))


def softmax_cross_entropy_loss(y, t):
    return -dot(t, log_softmax(y))


def softmax_cross_entropy_loss_gradient(y, t):
    return elements_sum(t) * softmax(y) - t


def softmax_cross_entropy_loss_gradient_one_hot(y, t):
    return softmax(y) - t


def Softmax_cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, log_softmax(Y)))


def Softmax_cross_entropy_loss_gradient(Y, T):
    N, K = Y.shape
    return hadamard(softmax(Y), column_repeat(rows_sum(T), K)) - T

def Softmax_cross_entropy_loss_gradient_one_hot(Y, T):
    return softmax(Y) - T


def stable_softmax_cross_entropy_loss(y, t):
    return -dot(t, stable_log_softmax(y))


def stable_softmax_cross_entropy_loss_gradient(y, t):
    return stable_softmax(y) * elements_sum(t) - t


def stable_softmax_cross_entropy_loss_gradient_one_hot(y, t):
    return stable_softmax(y) - t


def Stable_softmax_cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, stable_log_softmax(Y)))


def Stable_softmax_cross_entropy_loss_gradient(Y, T):
    N, K = Y.shape
    return hadamard(stable_softmax(Y), column_repeat(rows_sum(T), K)) - T


def Stable_softmax_cross_entropy_loss_gradient_one_hot(Y, T):
    return stable_softmax(Y) - T


def logistic_cross_entropy_loss(y, t):
    return -dot(t, log_sigmoid(y))


def logistic_cross_entropy_loss_gradient(y, t):
    return hadamard(t, Sigmoid(y)) - t


def Logistic_cross_entropy_loss(Y, T):
    return -elements_sum(hadamard(T, log_sigmoid(Y)))


def Logistic_cross_entropy_loss_gradient(Y, T):
    return hadamard(T, Sigmoid(Y)) - T


def negative_log_likelihood_loss(y, t):
    return -log(dot(y, t))


def negative_log_likelihood_loss_gradient(y, t):
    return -reciprocal(dot(y, t)) * t


def Negative_log_likelihood_loss(Y, T):
    return -elements_sum(log(rows_sum(hadamard(Y, T))))


def Negative_log_likelihood_loss_gradient(Y, T):
    N, K = Y.shape
    return -hadamard(column_repeat(reciprocal(rows_sum(hadamard(Y, T))), K), T)

Matrix = jnp.ndarray


class LossFunction(object):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        raise NotImplementedError

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        raise NotImplementedError


class SquaredErrorLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Squared_error_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Squared_error_loss_gradient(Y, T)


class MeanSquaredErrorLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Mean_squared_error_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Mean_squared_error_loss_gradient(Y, T)


class CrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Cross_entropy_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Cross_entropy_loss_gradient(Y, T)


class SoftmaxCrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Softmax_cross_entropy_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Softmax_cross_entropy_loss_gradient(Y, T)


class StableSoftmaxCrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Stable_softmax_cross_entropy_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Stable_softmax_cross_entropy_loss_gradient(Y, T)


class LogisticCrossEntropyLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Logistic_cross_entropy_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Logistic_cross_entropy_loss_gradient(Y, T)


class NegativeLogLikelihoodLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return Negative_log_likelihood_loss(Y, T)

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return Negative_log_likelihood_loss_gradient(Y, T)


def parse_loss_function(text: str) -> LossFunction:
    if text == "SquaredError":
        return SquaredErrorLossFunction()
    elif text == "MeanSquaredError":
        return MeanSquaredErrorLossFunction()
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