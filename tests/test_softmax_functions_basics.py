# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import unittest
from nerva_jax.matrix_operations import ones
from nerva_jax.softmax_functions import softmax, stable_softmax, log_softmax, exp, stable_log_softmax, rows_sum
from utilities import to_tensor, all_close, randn, all_finite, all_positive


class TestSoftmaxFunctionsBasics(unittest.TestCase):
    def test_softmax_rowwise_properties(self):
        X = to_tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        Y = softmax(X)
        # rows sum to 1
        self.assertTrue(all_close(rows_sum(Y), ones(X.shape[0]), atol=1e-6))
        # positive entries
        self.assertTrue(all_positive(Y))

    def test_softmax_invariance_to_shift(self):
        X = randn(4, 5)
        c = randn(4, 1)
        Y1 = softmax(X)
        Y2 = softmax(X + c)  # adding per-row constant shouldn't change result
        self.assertTrue(all_close(Y1, Y2, atol=1e-6))

    def test_stable_softmax_matches_on_moderate_values(self):
        X = randn(3, 4)
        self.assertTrue(all_close(softmax(X), stable_softmax(X), atol=1e-6))

    def test_log_softmax_relationship(self):
        X = randn(3, 4)
        Y = softmax(X)
        LS = log_softmax(X)
        self.assertTrue(all_close(exp(LS), Y, atol=1e-6))

    def test_stable_log_softmax_large_values(self):
        X = to_tensor([[1000.0, 1001.0, 1002.0]])
        # Should remain finite
        LS = stable_log_softmax(X)
        self.assertTrue(all_finite(LS))
        # Stable and naive should agree when subtracting max
        Xs = X - X.max(axis=1, keepdims=True)
        self.assertTrue(all_close(log_softmax(Xs), stable_log_softmax(X), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
