#!/bin/bash


git clone https://github.com/NVIDIA/cutlass.git

export CXX=g++
export CC=gcc

pip install torch
pip install setuptools
pip install ninja
pip install wheel

pip install .

#python test.py


# colab
ncu -f --target-processes all --set full \
--import-source on \
-o profile_flash_attn python test.py

# google cloud
#sudo -s
#/usr/local/cuda/bin/ncu -f --target-processes all --set full \
#                        --import-source on \
#                        -o profile_flash_attn python test.py