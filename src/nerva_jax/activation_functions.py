# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Activation functions and utilities used by the MLP implementation.

This module provides simple callable classes for common activations and a parser
that turns textual specifications into activation instances (e.g. "ReLU",
"LeakyReLU(alpha=0.1)", "SReLU(al=0, tl=0, ar=0, tr=1)").
"""

import jax.numpy as jnp

from nerva_jax.utilities import parse_function_call
from nerva_jax.matrix_operations import Matrix


def Relu(X: Matrix):
    """Rectified linear unit activation: max(0, X)."""
    return jnp.maximum(0, X)


def Relu_gradient(X: Matrix):
    """Gradient of ReLU: 1 where X > 0, 0 elsewhere."""
    return jnp.where(X > 0, 1, 0)


def Leaky_relu(alpha):
    """Leaky ReLU factory: max(X, alpha * X)."""
    return lambda X: jnp.maximum(alpha * X, X)


def Leaky_relu_gradient(alpha):
    """Gradient factory for leaky ReLU."""
    return lambda X: jnp.where(X > 0, 1, alpha)


def All_relu(alpha):
    """AllReLU factory."""
    return lambda X: jnp.where(X < 0, alpha * X, X)


def All_relu_gradient(alpha):
    """Gradient factory for AllReLU."""
    return lambda X: jnp.where(X < 0, alpha, 1)


def Hyperbolic_tangent(X: Matrix):
    """Hyperbolic tangent activation."""
    return jnp.tanh(X)


def Hyperbolic_tangent_gradient(X: Matrix):
    """Gradient of tanh: 1 - tanh²(X)."""
    return 1 - jnp.tanh(X) ** 2


def Sigmoid(X: Matrix):
    """Sigmoid activation: 1 / (1 + exp(-X))."""
    return 1 / (1 + jnp.exp(-X))


def Sigmoid_gradient(X: Matrix):
    """Gradient of sigmoid: σ(X) * (1 - σ(X))."""
    return Sigmoid(X) * (1 - Sigmoid(X))


def Srelu(al, tl, ar, tr):
    """SReLU factory: smooth rectified linear with learnable parameters."""
    return lambda X: jnp.where(X <= tl, tl + al * (X - tl),
                     jnp.where(X < tr, X, tr + ar * (X - tr)))


def Srelu_gradient(al, tl, ar, tr):
    """Gradient factory for SReLU."""
    return lambda X: jnp.where(X <= tl, al,
                     jnp.where(X < tr, 1, ar))


class ActivationFunction(object):
    """Interface for activation functions with value and gradient methods."""
    def __call__(self, X: Matrix) -> Matrix:
        raise NotImplementedError

    def gradient(self, X: Matrix) -> Matrix:
        raise NotImplementedError


class ReLUActivation(ActivationFunction):
    """ReLU activation function: max(0, x)."""
    def __call__(self, X: Matrix) -> Matrix:
        return Relu(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of ReLU."""
        return Relu_gradient(X)


class LeakyReLUActivation(ActivationFunction):
    """Leaky ReLU activation: max(x, alpha * x)."""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        """Apply leaky ReLU activation."""
        return Leaky_relu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of leaky ReLU."""
        return Leaky_relu_gradient(self.alpha)(X)


class AllReLUActivation(ActivationFunction):
    """AllReLU activation (alternative parameterization of leaky ReLU)."""
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        """Apply AllReLU activation."""
        return All_relu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of AllReLU."""
        return All_relu_gradient(self.alpha)(X)


class HyperbolicTangentActivation(ActivationFunction):
    """Hyperbolic tangent activation function."""
    def __call__(self, X: Matrix) -> Matrix:
        return Hyperbolic_tangent(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of hyperbolic tangent."""
        return Hyperbolic_tangent_gradient(X)


class SigmoidActivation(ActivationFunction):
    """Sigmoid activation function: 1 / (1 + exp(-x))."""
    def __call__(self, X: Matrix) -> Matrix:
        return Sigmoid(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of sigmoid."""
        return Sigmoid_gradient(X)


class SReLUActivation(ActivationFunction):
    """Smooth rectified linear activation with learnable parameters."""
    def __init__(self, al=0.0, tl=0.0, ar=0.0, tr=1.0):
        # Store the parameters and their gradients in matrices.
        # This is to make them usable for optimizers.
        self.x = jnp.array([al, tl, ar, tr])
        self.Dx = jnp.array([0.0, 0.0, 0.0, 0.0])

    def __call__(self, X: Matrix) -> Matrix:
        """Apply SReLU activation with current parameters."""
        al, tl, ar, tr = self.x
        return Srelu(al, tl, ar, tr)(X)

    def gradient(self, X: Matrix) -> Matrix:
        """Compute gradient of SReLU with current parameters."""
        al, tl, ar, tr = self.x
        return Srelu_gradient(al, tl, ar, tr)(X)


def parse_activation(text: str) -> ActivationFunction:
    """Parse a textual activation specification into an ActivationFunction.

    Examples include "ReLU", "Sigmoid", "HyperbolicTangent",
    "AllReLU(alpha=0.1)", "LeakyReLU(alpha=0.1)", and
    "SReLU(al=0, tl=0, ar=0, tr=1)".
    """
    try:
        func = parse_function_call(text)
        if func.name == 'ReLU':
            return ReLUActivation()
        elif func.name == 'Sigmoid':
            return SigmoidActivation()
        elif func.name == 'HyperbolicTangent':
            return HyperbolicTangentActivation()
        elif func.name == 'AllReLU':
            alpha = func.as_scalar('alpha')
            return AllReLUActivation(alpha)
        elif func.name == 'LeakyReLU':
            alpha = func.as_scalar('alpha')
            return LeakyReLUActivation(alpha)
        elif func.name == 'SReLU':
            al = func.as_scalar('al', 0)
            tl = func.as_scalar('tl', 0)
            ar = func.as_scalar('ar', 0)
            tr = func.as_scalar('tr', 1)
            return SReLUActivation(al, tl, ar, tr)
    except:
        pass
    raise RuntimeError(f'Could not parse activation "{text}"')
