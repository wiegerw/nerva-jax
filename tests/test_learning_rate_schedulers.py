# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import unittest
from nerva_jax.learning_rate import ConstantScheduler, TimeBasedScheduler, StepBasedScheduler, MultiStepLRScheduler, \
    ExponentialScheduler, parse_learning_rate


class TestLearningRateSchedulers(unittest.TestCase):
    def test_parse_learning_rate_valid(self):
        self.assertIsInstance(parse_learning_rate("Constant(0.1)"), ConstantScheduler)
        self.assertIsInstance(parse_learning_rate("TimeBased(0.1,0.5)"), TimeBasedScheduler)
        self.assertIsInstance(parse_learning_rate("StepBased(0.1,0.5,10)"), StepBasedScheduler)
        self.assertIsInstance(parse_learning_rate("MultiStepLR(0.1;1,3,5;0.2)"), MultiStepLRScheduler)
        self.assertIsInstance(parse_learning_rate("Exponential(0.1,0.2)"), ExponentialScheduler)

    def test_parse_learning_rate_invalid(self):
        with self.assertRaises(RuntimeError):
            parse_learning_rate("Unknown(0.1)")

    def test_constant_scheduler(self):
        s = ConstantScheduler(0.25)
        self.assertEqual(s(0), 0.25)
        self.assertEqual(s(10), 0.25)

    def test_time_based_scheduler(self):
        s = TimeBasedScheduler(1.0, 0.5)
        v0 = s(0)
        v1 = s(1)
        v2 = s(2)
        self.assertLess(v1, v0)
        self.assertLess(v2, v1)

    def test_step_based_scheduler(self):
        s = StepBasedScheduler(1.0, 0.5, 2)
        # floor((1+epoch)/2): epoch 0->0, 1->1, 2->1, 3->2
        self.assertAlmostEqual(s(0), 1.0)
        self.assertAlmostEqual(s(1), 0.5)
        self.assertAlmostEqual(s(2), 0.5)
        self.assertAlmostEqual(s(3), 0.25)

    def test_multistep_scheduler(self):
        s = MultiStepLRScheduler(1.0, [1,3,5], 0.1)
        self.assertAlmostEqual(s(0), 1.0)
        self.assertAlmostEqual(s(1), 0.1)
        self.assertAlmostEqual(s(2), 0.1)
        self.assertAlmostEqual(s(3), 0.01)

    def test_exponential_scheduler(self):
        s = ExponentialScheduler(2.0, 0.5)
        self.assertAlmostEqual(s(0), 2.0)
        self.assertAlmostEqual(s(1), 2.0 * 2.718281828459045 ** (-0.5), places=6)


if __name__ == '__main__':
    unittest.main()
