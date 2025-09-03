#!/usr/bin/env python3

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nerva_jax.activation_functions import ActivationFunction, HyperbolicTangentActivation
from nerva_jax.datasets import create_npz_dataloaders
from nerva_jax.layers import ActivationLayer, LinearLayer
from nerva_jax.learning_rate import TimeBasedScheduler
from nerva_jax.loss_functions import LossFunction
from nerva_jax.matrix_operations import elements_sum, Matrix
from nerva_jax.multilayer_perceptron import MultilayerPerceptron
from nerva_jax.optimizers import MomentumOptimizer, NesterovOptimizer, CompositeOptimizer
from nerva_jax.training import stochastic_gradient_descent
from nerva_jax.weight_initializers import zero_bias, xavier_normalized_weights

# ------------------------
# Custom activation function
# ------------------------

def Elu(alpha):
    return lambda X: jnp.where(X > 0, X, alpha * (jnp.exp(X) - 1))


def Elu_gradient(alpha):
    return lambda X: jnp.where(X > 0, jnp.ones_like(X), alpha * jnp.exp(X))


class ELUActivation(ActivationFunction):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, X: Matrix) -> Matrix:
        return Elu(self.alpha)(X)

    def gradient(self, X: Matrix) -> Matrix:
        return Elu_gradient(self.alpha)(X)


# ------------------------
# Custom weight initializer
# ------------------------

def lecun_weights(W: Matrix) -> Matrix:
    K, D = W.shape
    stddev = jnp.sqrt(1.0 / D)
    return np.random.randn(K, D) * stddev


# ------------------------
# Custom loss function
# ------------------------

class AbsoluteErrorLossFunction(LossFunction):
    def __call__(self, Y: Matrix, T: Matrix) -> float:
        return elements_sum(abs(Y - T))

    def gradient(self, Y: Matrix, T: Matrix) -> Matrix:
        return jnp.sign(Y - T)


def main():
    if not Path("../data/mnist-flattened.npz").exists():
        print("Error: MNIST dataset not found. Please provide the correct location or run the prepare_datasets.py script first.")
        return

    train_loader, test_loader = create_npz_dataloaders("../data/mnist-flattened.npz", batch_size=100)

    M = MultilayerPerceptron()

    # configure layer 1
    layer1 = ActivationLayer(784, 1024, ELUActivation(0.1))
    xavier_normalized_weights(layer1.W)
    zero_bias(layer1.b)
    optimizer_W = MomentumOptimizer(layer1, "W", "DW", 0.9)
    optimizer_b = NesterovOptimizer(layer1, "b", "Db", 0.75)
    layer1.optimizer = CompositeOptimizer([optimizer_W, optimizer_b])

    # configure layer 2
    layer2 = ActivationLayer(1024, 512, HyperbolicTangentActivation())
    layer1.W = lecun_weights(layer1.W)
    layer1.b = zero_bias(layer1.b)
    layer2.set_optimizer("Momentum(0.8)")

    # configure layer 3
    layer3 = LinearLayer(512, 10)
    layer3.set_weights("He")
    layer3.set_optimizer("GradientDescent")

    M.layers = [layer1, layer2, layer3]

    loss: LossFunction = AbsoluteErrorLossFunction()

    learning_rate = TimeBasedScheduler(lr=0.1, decay=0.09)

    epochs = 5

    stochastic_gradient_descent(M, epochs, loss, learning_rate, train_loader, test_loader)


if __name__ == '__main__':
    main()
