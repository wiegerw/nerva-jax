# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import os
import tempfile
import unittest
import numpy as np
from utilities import randn, to_long, equal_tensors
from nerva_jax.datasets import to_one_hot, from_one_hot, DataLoader, infer_num_classes, create_npz_dataloaders
from nerva_jax.matrix_operations import identity


class TestDatasetsBasics(unittest.TestCase):
    def test_one_hot_roundtrip(self):
        idx = to_long([0, 2, 1, 2])
        oh = to_one_hot(idx, num_classes=3)
        back = from_one_hot(oh)
        self.assertTrue(equal_tensors(back, idx))

    def test_memory_dataloader_batches_and_shapes(self):
        X = randn(10, 4)
        T = to_long([0, 1, 0, 2, 1, 2, 0, 1, 2, 2])
        loader = DataLoader(X, T, batch_size=3)  # num_classes inferred -> 4
        batches = list(iter(loader))
        self.assertEqual(len(batches), 4)
        for i, (Xi, Ti) in enumerate(batches):
            if i < 3:
                self.assertEqual(Xi.shape[0], 3)
                self.assertEqual(Ti.shape[0], 3)
            else:
                self.assertEqual(Xi.shape[0], 1)
                self.assertEqual(Ti.shape[0], 1)

    def test_infer_num_classes_indices_vs_onehot(self):
        Ttrain = to_long([0, 1, 2, 1])
        Ttest = to_long([2, 0, 1, 1])
        self.assertEqual(infer_num_classes(Ttrain, Ttest), 3)
        # one-hot
        Ttrain_oh = identity(4)[:3]  # 3x4 one-hot with width 4
        self.assertEqual(infer_num_classes(Ttrain_oh, Ttest), 4)

    def test_create_npz_dataloaders_roundtrip(self):
        # Tiny dataset with 6 samples
        Xtrain = np.random.randn(6, 3).astype(np.float32)
        Ttrain = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
        Xtest = np.random.randn(6, 3).astype(np.float32)
        Ttest = np.array([1, 0, 2, 2, 0, 1], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tmp_dataset.npz")
            np.savez(path, Xtrain=Xtrain, Ttrain=Ttrain, Xtest=Xtest, Ttest=Ttest)
            train_loader, test_loader = create_npz_dataloaders(path, batch_size=2)
            # loaders produce one-hot of expected width
            for Xb, Tb in train_loader:
                self.assertEqual(Xb.shape[1], 3)
                self.assertEqual(Tb.shape[1], 3)
            for Xb, Tb in test_loader:
                self.assertEqual(Xb.shape[1], 3)
                self.assertEqual(Tb.shape[1], 3)


if __name__ == '__main__':
    unittest.main()
