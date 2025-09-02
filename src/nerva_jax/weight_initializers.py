# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Weight and bias initialization helpers for linear layers."""

import jax.numpy as jnp
import numpy as np
from nerva_jax.matrix_operations import Matrix


def zero_bias(b: Matrix):
    """Zero bias."""
    return jnp.zeros_like(b)


def xavier_weights(W: Matrix):
    """Initialize weights using Xavier/Glorot initialization."""
    K, D = W.shape
    xavier_stddev = jnp.sqrt(2.0 / (K + D))
    return jnp.array(np.random.randn(K, D) * xavier_stddev)


def xavier_normalized_weights(W: Matrix):
    """Initialize weights using normalized Xavier initialization."""
    K, D = W.shape
    xavier_stddev = jnp.sqrt(2.0 / (K + D))
    return jnp.array(np.random.randn(K, D) * xavier_stddev)


def he_weights(W: Matrix):
    """Initialize weights using He initialization for ReLU networks."""
    K, D = W.shape
    he_stddev = jnp.sqrt(2.0 / D)
    random_matrix = np.random.randn(K, D)
    return jnp.array(random_matrix * he_stddev)


def set_layer_weights(layer, text: str):
    """Initialize a layer's parameters according to a named scheme."""
    if text == 'Xavier':
        layer.W = xavier_weights(layer.W)
        layer.b = zero_bias(layer.b)
    elif text == 'XavierNormalized':
        layer.W = xavier_normalized_weights(layer.W)
        layer.b = zero_bias(layer.b)
    elif text == 'He':
        layer.W = he_weights(layer.W)
        layer.b = zero_bias(layer.b)
    else:
        raise RuntimeError(f'Could not parse weight initializer "{text}"')
