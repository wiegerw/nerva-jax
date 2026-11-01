#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import unittest

import numpy as np

from nerva_jax.shapes_dataset import SHAPE_CLASSES, create_shapes_dataloaders, make_shapes, make_shapes_dataset
from utilities import to_numpy


class TestShapesDataset(unittest.TestCase):
    def test_make_shapes(self):
        X, T = make_shapes(200, seed=1)
        self.assertEqual(X.shape, (200, 784))
        self.assertEqual(X.dtype, np.float32)
        self.assertEqual(T.shape, (200,))
        self.assertTrue(np.all((0 <= X) & (X <= 1)))
        self.assertTrue(np.all((0 <= T) & (T < len(SHAPE_CLASSES))))
        self.assertEqual(len(np.unique(T)), len(SHAPE_CLASSES))

    def test_deterministic(self):
        X1, T1 = make_shapes(50, seed=7)
        X2, T2 = make_shapes(50, seed=7)
        X3, _ = make_shapes(50, seed=8)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(T1, T2)
        self.assertFalse(np.array_equal(X1, X3))

    def test_make_shapes_dataset(self):
        data = make_shapes_dataset(num_train=100, num_test=40, size=16, seed=3)
        self.assertEqual(data['Xtrain'].shape, (100, 256))
        self.assertEqual(data['Xtest'].shape, (40, 256))
        self.assertFalse(np.array_equal(data['Xtrain'][:40], data['Xtest']))

    def test_create_shapes_dataloaders(self):
        train_loader, test_loader = create_shapes_dataloaders(batch_size=10, num_train=50, num_test=20, seed=0)
        self.assertEqual(len(train_loader), 5)
        self.assertEqual(len(test_loader), 2)
        X, T = next(iter(train_loader))
        self.assertEqual(to_numpy(X).shape, (10, 784))
        self.assertEqual(to_numpy(T).shape, (10, len(SHAPE_CLASSES)))  # one-hot encoded


if __name__ == '__main__':
    unittest.main()
