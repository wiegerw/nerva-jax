@echo off

set PYTHONPATH=..\src

python -u ..\tools\create_datasets.py cifar10 --root=..\data
python -u ..\tools\create_datasets.py mnist --root=..\data
