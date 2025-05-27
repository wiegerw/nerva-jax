#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from pathlib import Path

from nerva_torch.activation_functions import ReLUActivation, LeakyReLUActivation
from nerva_torch.datasets import create_npz_dataloaders
from nerva_torch.layers import ActivationLayer, LinearLayer
from nerva_torch.learning_rate import TimeBasedScheduler
from nerva_torch.loss_functions import SoftmaxCrossEntropyLossFunction
from nerva_torch.multilayer_perceptron import MultilayerPerceptron
from nerva_torch.optimizers import MomentumOptimizer, NesterovOptimizer, CompositeOptimizer
from nerva_torch.training import sgd
from nerva_torch.weight_initializers import set_bias_to_zero, set_weights_xavier_normalized


def main():
    if not Path("../data/mnist-flattened.npz").exists():
        print("Error: MNIST dataset not found. Please provide the correct location or run the prepare_datasets.py script first.")
        return

    train_loader, test_loader = create_npz_dataloaders("../data/mnist-flattened.npz", batch_size=100)

    M = MultilayerPerceptron()

    # configure layer 1
    layer1 = ActivationLayer(784, 1024, ReLUActivation())
    set_weights_xavier_normalized(layer1.W)
    set_bias_to_zero(layer1.b)
    optimizer_W = MomentumOptimizer(layer1.W, layer1.DW, 0.9)
    optimizer_b = NesterovOptimizer(layer1.b, layer1.Db, 0.75)
    layer1.optimizer = CompositeOptimizer([optimizer_W, optimizer_b])

    # configure layer 2
    layer2 = ActivationLayer(1024, 512, LeakyReLUActivation(0.5))
    layer2.set_weights("Xavier")
    layer2.set_optimizer("Momentum(0.8)")

    # configure layer 3
    layer3 = LinearLayer(512, 10)
    layer3.set_weights("He")
    layer3.set_optimizer("GradientDescent")

    M.layers = [layer1, layer2, layer3]

    loss = SoftmaxCrossEntropyLossFunction()

    learning_rate = TimeBasedScheduler(lr=0.1, decay=0.09)

    epochs = 100

    sgd(M, epochs, loss, learning_rate, train_loader, test_loader)


if __name__ == '__main__':
    main()
