@echo off

set PYTHONPATH=..\src

if not exist ..\data\cifar10.npz (
    echo Error: file ..\data\cifar10.npz does not exist.
    echo Please provide the correct location or run the prepare_datasets.bat script first.
    exit /b 1
)

python -u ..\tools\mlp.py ^
    --layers="ReLU;ReLU;Linear" ^
    --sizes="3072,1024,512,10" ^
    --optimizers="Momentum(0.9);Momentum(0.9);Momentum(0.9)" ^
    --init-weights="Xavier,Xavier,Xavier" ^
    --batch-size=100 ^
    --epochs=1 ^
    --loss=SoftmaxCrossEntropy ^
    --learning-rate="Constant(0.01)" ^
    --dataset=..\data\cifar10.npz
