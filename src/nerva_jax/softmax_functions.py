# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import jax.numpy as jnp
from nerva_jax.matrix_operations import Diag, column_repeat, exp, hadamard, identity, is_row_vector, log, reciprocal, \
    row_repeat, rows_max, rows_sum

Matrix = jnp.ndarray

def softmax(X: Matrix) -> Matrix:
    N, D = X.shape
    E = exp(X)
    return hadamard(E, column_repeat(reciprocal(rows_sum(E)), D))


def softmax_jacobian(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    y = softmax(x)
    return Diag(y) - y.T * y


def stable_softmax(X: Matrix) -> Matrix:
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    E = exp(Y)
    return hadamard(E, column_repeat(reciprocal(rows_sum(E)), D))


def stable_softmax_jacobian(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    y = stable_softmax(x)
    return Diag(y) - y.T * y


def log_softmax(X: Matrix) -> Matrix:
    N, D = X.shape
    return X - column_repeat(log(rows_sum(exp(X))), D)


def log_softmax_jacobian(x: Matrix) -> Matrix:
    assert is_row_vector(x)
    N, D = x.shape
    return identity(D) - row_repeat(softmax(x), D)


def stable_log_softmax(X: Matrix) -> Matrix:
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    return Y - column_repeat(log(rows_sum(exp(Y))), D)


def stable_log_softmax_jacobian(x: Matrix) -> Matrix:
    return log_softmax_jacobian(x)

Matrix = jnp.ndarray

class SoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return softmax(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return softmax_jacobian(X)


class StableSoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return stable_softmax(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return stable_softmax_jacobian(X)


class LogSoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return log_softmax(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return log_softmax_jacobian(X)


class StableLogSoftmaxFunction(object):
    def __call__(self, X: Matrix) -> Matrix:
        return stable_log_softmax(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return stable_log_softmax_jacobian(X)