#!/bin/sh

# Copyright 2025 Wieger Wesselink.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE or http://www.boost.org/LICENSE_1_0.txt)

cd "$(dirname "$0")"
export PYTHONPATH=../src
dataset=shapes.npz

# Generate the synthetic shapes dataset (28x28 images of 10 kinds of shapes) if it does not exist yet
if [ ! -f $dataset ]; then
    python3 -m nerva_jax.shapes_dataset --output=$dataset || exit 1
fi

python3 -u ../tools/mlp.py \
        --layers="ReLU;ReLU;Linear" \
        --layer-sizes="784;1024;512;10" \
        --layer-weights="XavierNormal;XavierNormal;XavierNormal" \
        --optimizers="Momentum(0.9);Momentum(0.9);Momentum(0.9)" \
        --batch-size=100 \
        --epochs=5 \
        --loss=SoftmaxCrossEntropy \
        --learning-rate="Constant(0.01)" \
        --load-dataset=$dataset
