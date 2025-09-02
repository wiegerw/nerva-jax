# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import unittest
from nerva_jax.optimizers import GradientDescentOptimizer, MomentumOptimizer, NesterovOptimizer, CompositeOptimizer, \
    parse_optimizer
from utilities import to_tensor, all_close


class DummyObj:
    """Minimal object to hold x and Dx as attributes."""
    def __init__(self, x, Dx):
        self.x = x
        self.Dx = Dx


class TestOptimizersBasics(unittest.TestCase):
    def test_parse_optimizer_valid(self):
        self.assertTrue(callable(parse_optimizer("GradientDescent")))
        self.assertTrue(callable(parse_optimizer("Momentum(mu=0.9)")))
        self.assertTrue(callable(parse_optimizer("Nesterov(mu=0.9)")))

    def test_parse_optimizer_invalid(self):
        with self.assertRaises(RuntimeError):
            parse_optimizer("Unknown()")

    def test_gradient_descent_update(self):
        obj = DummyObj(to_tensor([1.0, -2.0]), to_tensor([0.5, -1.0]))
        opt = GradientDescentOptimizer(obj, "x", "Dx")
        opt.update(eta=0.2)
        self.assertTrue(all_close(obj.x, to_tensor([0.9, -1.8])))

    def test_momentum_update(self):
        obj = DummyObj(to_tensor([0.0, 0.0]), to_tensor([1.0, -2.0]))
        opt = MomentumOptimizer(obj, "x", "Dx", mu=0.9)
        opt.update(eta=0.1)  # delta_x = [-0.1, 0.2]
        self.assertTrue(all_close(obj.x, to_tensor([-0.1, 0.2])))
        opt.update(eta=0.1)  # delta_x = [-0.19, 0.38]
        self.assertTrue(all_close(obj.x, to_tensor([-0.29, 0.58])))

    def test_nesterov_update(self):
        obj = DummyObj(to_tensor([0.0, 0.0]), to_tensor([1.0, -2.0]))
        opt = NesterovOptimizer(obj, "x", "Dx", mu=0.9)
        opt.update(eta=0.1)
        self.assertTrue(all_close(obj.x, to_tensor([-0.19, 0.38])))

    def test_composite_optimizer(self):
        obj1 = DummyObj(to_tensor([1.0]), to_tensor([2.0]))
        obj2 = DummyObj(to_tensor([3.0]), to_tensor([4.0]))
        o1 = GradientDescentOptimizer(obj1, "x", "Dx")
        o2 = GradientDescentOptimizer(obj2, "x", "Dx")
        comp = CompositeOptimizer([o1, o2])
        comp.update(eta=0.5)
        self.assertTrue(all_close(obj1.x, to_tensor([0.0])))
        self.assertTrue(all_close(obj2.x, to_tensor([1.0])))


if __name__ == '__main__':
    unittest.main()
