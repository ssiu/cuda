#!/bin/bash


git clone https://github.com/NVIDIA/cutlass.git

export CXX=g++
export CC=gcc

pip install torch
pip install setuptools
pip install ninja
pip install wheel

pip install .

#python benchmark_flash.py



# colab
#ncu -f --target-processes all --set full \
#--import-source on \
#-o profile_flash_attn python benchmark_flash.py

#for size in "${sizes[@]}"; do
#    echo "Running with size $size..."
#    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
#        --csv ./launch_sm75_gemm.o "$size" "$size" "$size" > "${size}.csv"

sizes=(512 1024 2048 4096 8192 16384)

for size in "${sizes[@]}"; do
    echo "Running with size $size..."

    ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --csv python benchmark_flash.py 4 "$size" 32 128 > "${size}.csv"

    echo "Output saved to ${size}.csv"

done

#ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed python benchmark_flash.py

# google cloud
#sudo -s
#/usr/local/cuda/bin/ncu -f --target-processes all --set full \
#                        --import-source on \
#                        -o profile_flash_attn python benchmark_flash.py

#/usr/local/cuda/bin/ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv python benchmark_flash.py > "profile.csv"
#sudo su
#/usr/local/cuda/bin/ncu --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed python benchmark_flash.py
