# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Weight and bias initialization helpers for linear layers."""

import math
import jax.numpy as jnp
import numpy as np
from nerva_jax.utilities import parse_function_call
from nerva_jax.matrix_operations import Matrix


def bias_uniform(b_: Matrix, a: float = 0.0, b: float = 1.0):
    """Uniform initialization within [a, b)."""
    return jnp.asarray(np.random.uniform(a, b, size=b_.shape))


def bias_normal(b: Matrix, mean: float = 0.0, std: float = 1.0):
    """Normal (Gaussian) initialization with given mean and std."""
    return jnp.asarray(np.random.normal(mean, std, size=b.shape))


def bias_zero(b: Matrix):
    """Initialize biases to zero."""
    return jnp.zeros_like(b)


def weights_uniform(W: Matrix, a: float = 0.0, b: float = 1.0):
    """Uniform initialization within [a, b)."""
    return jnp.asarray(np.random.uniform(a, b, size=W.shape))


def weights_normal(W: Matrix, mean: float = 0.0, std: float = 1.0):
    """Normal (Gaussian) initialization with given mean and std."""
    return jnp.asarray(np.random.normal(mean, std, size=W.shape))


def weights_zero(W: Matrix):
    """Initialize weights to zero."""
    return jnp.zeros_like(W)


def weights_xavier_uniform(W: Matrix):
    """Xavier / Glorot uniform initialization (for tanh/sigmoid).

    K = fan-out (output size)
    D = fan-in  (input size)
    """
    K, D = W.shape
    limit = math.sqrt(6.0 / (D + K))  # sqrt(6 / (fan_in + fan_out))
    return jnp.asarray(np.random.uniform(-limit, limit, size=W.shape))


def weights_xavier_normal(W: Matrix):
    """Xavier / Glorot normal initialization (for tanh/sigmoid).

    K = fan-out (output size)
    D = fan-in  (input size)
    """
    K, D = W.shape
    std = math.sqrt(2.0 / (D + K))  # sqrt(2 / (fan_in + fan_out))
    return jnp.asarray(np.random.normal(0.0, std, size=W.shape))


def weights_he_normal(W: Matrix):
    """He / Kaiming normal initialization (for ReLU).

    K = fan-out (output size)
    D = fan-in  (input size)
    """
    K, D = W.shape
    std = math.sqrt(2.0 / D)  # sqrt(2 / fan_in)
    return jnp.asarray(np.random.normal(0.0, std, size=W.shape))


def weights_he_uniform(W: Matrix):
    """He / Kaiming uniform initialization (for ReLU, less common).

    K = fan-out (output size)
    D = fan-in  (input size)
    """
    K, D = W.shape
    limit = math.sqrt(6.0 / D)  # sqrt(6 / fan_in)
    return jnp.asarray(np.random.uniform(-limit, limit, size=W.shape))


def set_layer_weights(layer, text: str):
    """Initialize a layer's parameters according to a named scheme."""
    func = parse_function_call(text)
    if func.name == 'Uniform':
        a = func.as_scalar('a', 0)
        b = func.as_scalar('b', 1)
        layer.W = weights_uniform(layer.W, a, b)
        layer.b = bias_zero(layer.b)
    elif func.name == 'Normal':
        a = func.as_scalar('a', 0)
        b = func.as_scalar('b', 1)
        layer.W = weights_normal(layer.W, a, b)
        layer.b = bias_zero(layer.b)
    if func.name == 'XavierUniform':
        layer.W = weights_xavier_uniform(layer.W)
        layer.b = bias_zero(layer.b)
    elif func.name == 'XavierNormal':
        layer.W = weights_xavier_normal(layer.W)
        layer.b = bias_zero(layer.b)
    elif func.name == 'HeUniform':
        layer.W = weights_he_uniform(layer.W)
        layer.b = bias_zero(layer.b)
    elif func.name == 'HeNormal':
        layer.W = weights_he_normal(layer.W)
        layer.b = bias_zero(layer.b)
    elif func.name == 'Zero':
        layer.W = weights_zero(layer.W)
        layer.b = bias_zero(layer.b)
    else:
        raise RuntimeError(f'Could not parse weight initializer "{text}"')
