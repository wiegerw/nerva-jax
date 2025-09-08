#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from pathlib import Path

from nerva_jax.activation_functions import ReLUActivation, LeakyReLUActivation
from nerva_jax.datasets import create_npz_dataloaders
from nerva_jax.layers import ActivationLayer, LinearLayer
from nerva_jax.learning_rate import TimeBasedScheduler
from nerva_jax.loss_functions import StableSoftmaxCrossEntropyLossFunction
from nerva_jax.multilayer_perceptron import MultilayerPerceptron, parse_multilayer_perceptron
from nerva_jax.optimizers import MomentumOptimizer, NesterovOptimizer, CompositeOptimizer
from nerva_jax.training import stochastic_gradient_descent
from nerva_jax.weight_initializers import bias_zero, weights_xavier_normal


def construct_mlp1() -> MultilayerPerceptron:
    # tag::construct1[]
    M = MultilayerPerceptron()

    # configure layer 1
    layer1 = ActivationLayer(784, 1024, ReLUActivation())
    layer1.W = weights_xavier_normal(layer1.W)
    layer1.b = bias_zero(layer1.b)
    optimizer_W = MomentumOptimizer(layer1, "W", "DW", 0.9)
    optimizer_b = NesterovOptimizer(layer1, "b", "Db", 0.75)
    layer1.optimizer = CompositeOptimizer([optimizer_W, optimizer_b])

    # configure layer 2
    layer2 = ActivationLayer(1024, 512, LeakyReLUActivation(0.5))
    layer2.set_weights("XavierNormal")
    layer2.set_optimizer("Momentum(0.8)")

    # configure layer 3
    layer3 = LinearLayer(512, 10)
    layer3.set_weights("HeNormal")
    layer3.set_optimizer("GradientDescent")

    M.layers = [layer1, layer2, layer3]
    # end::construct1[]

    return M


def construct_mlp2() -> MultilayerPerceptron:
    # tag::construct2[]
    layer_specifications = ["ReLU", "LeakyReLU(0.5)", "Linear"]
    linear_layer_sizes = [784, 1024, 512, 10]
    linear_layer_optimizers = ["Nesterov(0.9)", "Momentum(0.8)", "GradientDescent"]
    linear_layer_weight_initializers = ["XavierNormal", "XavierUniform", "HeNormal"]
    M = parse_multilayer_perceptron(layer_specifications,
                                    linear_layer_sizes,
                                    linear_layer_optimizers,
                                    linear_layer_weight_initializers)
    # end::construct2[]
    return M


def main():
    if not Path("../data/mnist-flattened.npz").exists():
        print("Error: MNIST dataset not found. Please provide the correct location or run the prepare_datasets.py script first.")
        return

    train_loader, test_loader = create_npz_dataloaders("../data/mnist-flattened.npz", batch_size=100)

    loss = StableSoftmaxCrossEntropyLossFunction()
    learning_rate = TimeBasedScheduler(lr=0.1, decay=0.09)
    epochs = 5

    M1 = construct_mlp1()
    stochastic_gradient_descent(M1, epochs, loss, learning_rate, train_loader, test_loader)

    M2 = construct_mlp2()
    stochastic_gradient_descent(M2, epochs, loss, learning_rate, train_loader, test_loader)


if __name__ == '__main__':
    main()
