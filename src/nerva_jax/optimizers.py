# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from typing import Any, Callable, List

import jax.numpy as jnp
from nerva_jax.utilities import parse_function_call


class Optimizer(object):
    def update(self, eta):
        raise NotImplementedError


class CompositeOptimizer(Optimizer):
    def __init__(self, optimizers: List[Optimizer]):
        self.optimizers = optimizers

    def update(self, eta):
        for optimizer in self.optimizers:
            optimizer.update(eta)


class GradientDescentOptimizer(Optimizer):
    def __init__(self, obj, attr_x: str, attr_Dx: str):
        """
        Store the names of the x and Dx attributes
        """
        self.obj = obj
        self.attr_x = attr_x
        self.attr_Dx = attr_Dx

    def update(self, eta):
        x = getattr(self.obj, self.attr_x)
        Dx = getattr(self.obj, self.attr_Dx)
        x1 = x - eta * Dx
        setattr(self.obj, self.attr_x, x1)


class MomentumOptimizer(GradientDescentOptimizer):
    def __init__(self, obj, attr_x: str, attr_Dx: str, mu: float):
        super().__init__(obj, attr_x, attr_Dx)
        self.mu = mu
        x = getattr(self.obj, self.attr_x)
        self.delta_x = jnp.zeros_like(x)

    def update(self, eta):
        x = getattr(self.obj, self.attr_x)
        Dx = getattr(self.obj, self.attr_Dx)
        self.delta_x = self.mu * self.delta_x - eta * Dx
        x1 = x + self.delta_x
        setattr(self.obj, self.attr_x, x1)

class NesterovOptimizer(MomentumOptimizer):
    def __init__(self, obj, attr_x: str, attr_Dx: str, mu: float):
        super().__init__(obj, attr_x, attr_Dx, mu)

    def update(self, eta):
        x = getattr(self.obj, self.attr_x)
        Dx = getattr(self.obj, self.attr_Dx)
        self.delta_x_prev = self.delta_x
        self.delta_x = self.mu * self.delta_x - eta * Dx
        x1 = x + self.mu * self.delta_x - eta * Dx
        setattr(self.obj, self.attr_x, x1)


def parse_optimizer(text: str) -> Callable[[Any, Any], Optimizer]:
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
