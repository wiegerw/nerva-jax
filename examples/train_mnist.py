#!/usr/bin/env python3

# Copyright 2023 - 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

from pathlib import Path

from nerva_jax.activation_functions import ReLUActivation
from nerva_jax.datasets import create_npz_dataloaders, DataLoader
from nerva_jax.layers import ActivationLayer, LinearLayer
from nerva_jax.learning_rate import ConstantScheduler, LearningRateScheduler
from nerva_jax.loss_functions import SoftmaxCrossEntropyLossFunction, LossFunction
from nerva_jax.multilayer_perceptron import MultilayerPerceptron
from nerva_jax.training import compute_statistics
from nerva_jax.utilities import StopWatch


# The Core Training Loop (Mini-batch SGD)
def sgd(M: MultilayerPerceptron,
        epochs: int,
        loss: LossFunction,
        learning_rate: LearningRateScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader
       ):

    training_time = 0.0

    for epoch in range(epochs):
        timer = StopWatch()

        # Get the learning rate for the current epoch
        lr = learning_rate(epoch)

        # Iterate over mini-batches
        for (X, T) in train_loader:
            # 1. Feedforward: Pass input data through the model
            Y = M.feedforward(X)

            # 2. Compute Loss Gradient: Calculate the gradient of the loss
            #    with respect to the model's output Y.
            #    Divide by batch size for average gradient.
            DY = loss.gradient(Y, T) / Y.shape[0]

            # 3. Backpropagate: Compute gradients of the loss
            #    with respect to layer weights and biases.
            M.backpropagate(Y, DY)

            # 4. Optimize: Update layer weights and biases
            #    using the calculated gradients and the optimizer.
            M.optimize(lr)

        seconds = timer.seconds()
        training_time += seconds
        compute_statistics(M, lr, loss, train_loader, test_loader, epoch=epoch + 1, elapsed_seconds=seconds)

    print(f'Total training time for the {epochs} epochs: {training_time:.8f}s\n')


def main():
    if not Path("../data/mnist-flattened.npz").exists():
        print("Error: MNIST dataset not found. Please provide the correct location or run the prepare_datasets.py script first.")
        return

    train_loader, test_loader = create_npz_dataloaders("../data/mnist-flattened.npz", batch_size=100)

    # Create a new MLP model
    M = MultilayerPerceptron()

    # Define and add layers
    M.layers = [
        # Input layer followed by a ReLU activation
        ActivationLayer(784, 1024, ReLUActivation()),

        # Another layer with ReLU activation
        ActivationLayer(1024, 512, ReLUActivation()),

        # Output layer (Linear for classification scores)
        LinearLayer(512, 10)
    ]

    for layer in M.layers:
        # Set the optimizer for the layer
        layer.set_optimizer('Momentum(0.9)')

        # Set the weight initialization method
        layer.set_weights('Xavier')

    # Define the loss function
    loss = SoftmaxCrossEntropyLossFunction()

    # Define the learning rate scheduler (e.g., a constant learning rate of 0.01)
    learning_rate = ConstantScheduler(0.01)

    # Define the number of training epochs
    epochs = 5

    sgd(M, epochs, loss, learning_rate, train_loader, test_loader)


if __name__ == '__main__':
    main()
