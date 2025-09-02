# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Softmax and log-softmax functions together with stable variants.

This module provides both function-only forms and simple callable classes.
"""

import jax.numpy as jnp
from nerva_jax.matrix_operations import Diag, column_repeat, exp, hadamard, identity, is_row_vector, log, reciprocal, \
    row_repeat, rows_max, rows_sum, Matrix


def softmax(X: Matrix) -> Matrix:
    """Row-wise softmax with explicit normalization (numerically unsafe)."""
    N, D = X.shape
    E = exp(X)
    return hadamard(E, column_repeat(reciprocal(rows_sum(E)), D))


def softmax_jacobian(x: Matrix) -> Matrix:
    """Jacobian matrix of softmax for a single row vector."""
    assert is_row_vector(x)
    y = softmax(x)
    return Diag(y) - y.T * y


def stable_softmax(X: Matrix) -> Matrix:
    """Row-wise softmax using max-subtraction for numerical stability."""
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    E = exp(Y)
    return hadamard(E, column_repeat(reciprocal(rows_sum(E)), D))


def stable_softmax_jacobian(x: Matrix) -> Matrix:
    """Jacobian matrix of stable softmax for a single row vector."""
    assert is_row_vector(x)
    y = stable_softmax(x)
    return Diag(y) - y.T * y


def log_softmax(X: Matrix) -> Matrix:
    """Row-wise log-softmax (numerically unsafe version)."""
    N, D = X.shape
    return X - column_repeat(log(rows_sum(exp(X))), D)


def log_softmax_jacobian(x: Matrix) -> Matrix:
    """Jacobian matrix of log_softmax for a single row vector."""
    assert is_row_vector(x)
    N, D = x.shape
    return identity(D) - row_repeat(softmax(x), D)


def stable_log_softmax(X: Matrix) -> Matrix:
    """Row-wise log-softmax with max-subtraction for stability."""
    N, D = X.shape
    Y = X - column_repeat(rows_max(X), D)
    return Y - column_repeat(log(rows_sum(exp(Y))), D)


def stable_log_softmax_jacobian(x: Matrix) -> Matrix:
    """Jacobian matrix of stable log_softmax (same as log_softmax)."""
    return log_softmax_jacobian(x)


class SoftmaxFunction(object):
    """Callable implementing row-wise softmax and its Jacobian."""

    def __call__(self, X: Matrix) -> Matrix:
        return softmax(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return softmax_jacobian(X)


class StableSoftmaxFunction(object):
    """Callable implementing numerically stable row-wise softmax and its Jacobian."""

    def __call__(self, X: Matrix) -> Matrix:
        return stable_softmax(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return stable_softmax_jacobian(X)


class LogSoftmaxFunction(object):
    """Callable implementing row-wise log-softmax and its Jacobian."""

    def __call__(self, X: Matrix) -> Matrix:
        return log_softmax(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return log_softmax_jacobian(X)


class StableLogSoftmaxFunction(object):
    """Callable implementing numerically stable row-wise log-softmax and its Jacobian."""

    def __call__(self, X: Matrix) -> Matrix:
        return stable_log_softmax(X)

    def jacobian(self, X: Matrix) -> Matrix:
        return stable_log_softmax_jacobian(X)
