#!/usr/bin/env python3

# Copyright 2023 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

import argparse

from nerva_jax.training import train


def make_argument_parser():
    cmdline_parser = argparse.ArgumentParser()

    # model parameters
    cmdline_parser.add_argument('--layers', type=str, required=True, help='A semi-colon separated lists of layer specifications.')
    cmdline_parser.add_argument('--layer-sizes', type=str, default='3072;128;64;10', help='A semi-colon separated list of layer sizes, e.g. "3072;128;64;10".')
    cmdline_parser.add_argument('--layer-weights', type=str, required=True, help='A semi-colon separated list of weight initializers, one per linear layer (e.g. "XavierNormal;XavierNormal;XavierNormal"). A single value is used for all linear layers.')

    # learning rate
    cmdline_parser.add_argument("--learning-rate", type=str, required=True, help="The learning rate scheduler")

    # loss function
    cmdline_parser.add_argument('--loss', type=str, required=True, help='The loss function')

    # training parameters
    cmdline_parser.add_argument("--epochs", help="The number of epochs", type=int, default=100)
    cmdline_parser.add_argument("--batch-size", help="The batch size", type=int, default=1)

    # optimizer
    cmdline_parser.add_argument("--optimizers", type=str, help="A semi-colon separated list of optimizers, one per layer (GradientDescent, Momentum(<mu>), Nesterov(<mu>)). A single value is used for all layers.", default="GradientDescent")

    # dataset
    cmdline_parser.add_argument('--load-dataset', type=str, required=True, help='The .npz file containing train and test data')

    # weights
    cmdline_parser.add_argument('--load-weights', type=str, help='Load model weights and biases from a .npz file.')

    # randomness
    cmdline_parser.add_argument("--seed", type=int, help="Seed for the random number generators (weight initialization)")

    # logging
    cmdline_parser.add_argument("--debug", help="Log intermediate values", action="store_true")

    return cmdline_parser


def main():
    cmdline_parser = make_argument_parser()
    args = cmdline_parser.parse_args()

    linear_layer_sizes = [int(s) for s in args.layer_sizes.split(';')]
    layer_specifications = args.layers.split(';')
    linear_layer_weight_initializers = args.layer_weights.split(';')
    linear_layer_optimizers = args.optimizers.split(';')

    # A single weight initializer or optimizer applies to all layers
    if len(linear_layer_weight_initializers) == 1:
        linear_layer_weight_initializers *= len(linear_layer_sizes) - 1
    if len(linear_layer_optimizers) == 1:
        linear_layer_optimizers *= len(layer_specifications)

    train(layer_specifications,
          linear_layer_sizes,
          linear_layer_optimizers,
          linear_layer_weight_initializers,
          args.batch_size,
          args.epochs,
          args.loss,
          args.learning_rate,
          args.load_weights,
          args.load_dataset,
          args.debug,
          args.seed
         )

if __name__ == '__main__':
    main()
