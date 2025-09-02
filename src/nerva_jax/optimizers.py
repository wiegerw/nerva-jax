# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

"""Optimizers used to adjusts the model's parameters based on the gradients.

   Only SGD, Momentum and Nesterov variants are provided. The parser creates
   factory callables from textual specifications like "Momentum(mu=0.9)".
"""

from typing import Any, Callable, List

import jax.numpy as jnp
from nerva_jax.utilities import parse_function_call


class Optimizer(object):
    """Minimal optimizer interface used by layers to update parameters."""
    def update(self, eta):
        raise NotImplementedError


class CompositeOptimizer(Optimizer):
    """Combines multiple optimizers to update different parameter groups."""
    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers

    def update(self, eta):
        """Update all contained optimizers with the given learning rate."""
        for optimizer in self.optimizers:
            optimizer.update(eta)


class GradientDescentOptimizer(Optimizer):
    """Standard gradient descent optimizer: x -= eta * grad."""
    def __init__(self, obj, attr_x: str, attr_Dx: str):
        """
        Store the names of the x and Dx attributes
        """
        self.obj = obj
        self.attr_x = attr_x
        self.attr_Dx = attr_Dx

    def update(self, eta):
        """Apply gradient descent update step."""
        x = getattr(self.obj, self.attr_x)
        Dx = getattr(self.obj, self.attr_Dx)
        x1 = x - eta * Dx
        setattr(self.obj, self.attr_x, x1)


class MomentumOptimizer(GradientDescentOptimizer):
    """Gradient descent with momentum for accelerated convergence."""
    def __init__(self, obj, attr_x: str, attr_Dx: str, mu: float):
        super().__init__(obj, attr_x, attr_Dx)
        self.mu = mu
        x = getattr(self.obj, self.attr_x)
        self.delta_x = jnp.zeros_like(x)

    def update(self, eta):
        """Apply momentum update step."""
        x = getattr(self.obj, self.attr_x)
        Dx = getattr(self.obj, self.attr_Dx)
        self.delta_x = self.mu * self.delta_x - eta * Dx
        x1 = x + self.delta_x
        setattr(self.obj, self.attr_x, x1)

class NesterovOptimizer(MomentumOptimizer):
    """Nesterov accelerated gradient descent optimizer."""
    def __init__(self, obj, attr_x: str, attr_Dx: str, mu: float):
        super().__init__(obj, attr_x, attr_Dx, mu)

    def update(self, eta):
        """Apply Nesterov accelerated gradient update step."""
        x = getattr(self.obj, self.attr_x)
        Dx = getattr(self.obj, self.attr_Dx)
        self.delta_x_prev = self.delta_x
        self.delta_x = self.mu * self.delta_x - eta * Dx
        x1 = x + self.mu * self.delta_x - eta * Dx
        setattr(self.obj, self.attr_x, x1)


def parse_optimizer(text: str) -> Callable[[Any, str, str], Optimizer]:
    """Parse a textual optimizer specification into a factory function.

    Returns a callable that takes (x, Dx) and produces an Optimizer.
    Supported names: GradientDescent, Momentum(mu=...), Nesterov(mu=...).
    """
    try:
        func = parse_function_call(text)
        if func.name == 'GradientDescent':
            return lambda obj, attr_x, attr_Dx: GradientDescentOptimizer(obj, attr_x, attr_Dx)
        elif func.name == 'Momentum':
            mu = func.as_scalar('mu')
            return lambda obj, attr_x, attr_Dx: MomentumOptimizer(obj, attr_x, attr_Dx, mu)
        elif func.name == 'Nesterov':
            mu = func.as_scalar('mu')
            return lambda obj, attr_x, attr_Dx: NesterovOptimizer(obj, attr_x, attr_Dx, mu)
    except:
        pass
    raise RuntimeError(f'Could not parse optimizer "{text}"')
