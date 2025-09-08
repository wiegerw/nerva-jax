#!/bin/sh

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

PYTHONPATH=../src
dataset=../data/cifar10-flattened.npz

if [ ! -f $dataset ]; then
    echo "Error: file $dataset does not exist."
    echo "Please provide the correct location or run the prepare_datasets.py script first."
    exit 1
fi

# tag::doc[]
python3 -u ../tools/mlp.py \
        --layers="ReLU;ReLU;Linear" \
        --layer-sizes="3072;1024;512;10" \
        --layer-weights="XavierNormal;XavierNormal;XavierNormal" \
        --optimizers="Momentum(0.9);Momentum(0.9);Momentum(0.9)" \
        --batch-size=100 \
        --epochs=5 \
        --loss=SoftmaxCrossEntropy \
        --learning-rate="Constant(0.01)" \
        --load-dataset=$dataset
# end::doc[]
