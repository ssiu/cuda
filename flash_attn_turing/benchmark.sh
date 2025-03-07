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
#ncu -f --target-processes all --set full \
#--import-source on \
#-o profile_flash_attn python test.py

#for size in "${sizes[@]}"; do
#    echo "Running with size $size..."
#    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
#        --csv ./launch_sm75_gemm.o "$size" "$size" "$size" > "${size}.csv"

# gpu__time_duration.sum
# sm__throughput.avg.pct_of_peak_sustained_elapsed

# ! ncu --metrics gpu__time_duration.sum --csv python test.py > "profile.csv"

#ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed --csv python test.py > "profile.csv"
ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv python test.py > "profile.csv"

# google cloud
#sudo -s
#/usr/local/cuda/bin/ncu -f --target-processes all --set full \
#                        --import-source on \
#                        -o profile_flash_attn python test.py


