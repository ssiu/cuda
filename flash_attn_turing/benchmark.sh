#!/bin/bash


git clone https://github.com/NVIDIA/cutlass.git

export CXX=g++
export CC=gcc

pip install .

python test.py

#ncu -f --target-processes all --set full \
#--import-source on \
#-o profile_flash_attn python test.py