# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import unittest
from utilities import to_tensor, all_close, all_true, all_finite
from nerva_jax.matrix_operations import zeros, ones, identity, product, hadamard, columns_sum, \
    rows_sum, columns_mean, rows_mean, columns_max, rows_max, inv_sqrt


class TestMatrixOperationsBasics(unittest.TestCase):
    def test_zeros_ones_identity(self):
        Z = zeros(2, 3)
        O = ones(2, 3)
        I = identity(3)
        self.assertEqual(Z.shape, (2, 3))
        self.assertTrue(all_true(Z == 0))
        self.assertEqual(O.shape, (2, 3))
        self.assertTrue(all_true(O == 1))
        self.assertEqual(I.shape, (3, 3))

    def test_product_and_hadamard(self):
        X = to_tensor([[1.0, 2.0], [3.0, 4.0]])
        Y = to_tensor([[5.0, 6.0], [7.0, 8.0]])
        self.assertTrue(all_close(product(X, Y), X @ Y))
        self.assertTrue(all_close(hadamard(X, Y), X * Y))

    def test_sums_means_max(self):
        X = to_tensor([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
        self.assertTrue(all_close(columns_sum(X), X.sum(axis=0)))
        self.assertTrue(all_close(rows_sum(X), X.sum(axis=1)))
        self.assertTrue(all_close(columns_mean(X), X.mean(axis=0)))
        self.assertTrue(all_close(rows_mean(X), X.mean(axis=1)))
        self.assertTrue(all_close(columns_max(X), X.max(axis=0)))
        self.assertTrue(all_close(rows_max(X), X.max(axis=1)))

    def test_inv_sqrt_stability_and_log_sigmoid(self):
        X = to_tensor([0.0, 1.0, 4.0])
        inv = inv_sqrt(X)
        # Finite values due to epsilon
        self.assertTrue(all_finite(inv))


if __name__ == '__main__':
    unittest.main()
