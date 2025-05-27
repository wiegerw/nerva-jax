#!/bin/sh

PYTHONPATH=../src
dataset=../data/cifar10-flattened.npz

if [ ! -f $dataset ]; then
    echo "Error: file ../data/mnist.npz does not exist."
    echo "Please provide the correct location or run the prepare_datasets.sh script first."
    exit 1
fi

python3 -u ../tools/mlp.py \
        --layers="ReLU;ReLU;Linear" \
        --sizes="3072,1024,512,10" \
        --optimizers="Momentum(0.9);Momentum(0.9);Momentum(0.9)" \
        --init-weights="Xavier,Xavier,Xavier" \
        --batch-size=100 \
        --epochs=1 \
        --loss=SoftmaxCrossEntropy \
        --learning-rate="Constant(0.01)" \
        --dataset=$dataset
