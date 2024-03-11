#!/bin/sh

PYTHONPATH=../src

python3 -u ../tools/create_datasets.py cifar10 --root=../data
python3 -u ../tools/create_datasets.py mnist --root=../data
