#!/bin/bash

for file in $(ls | grep _mm.cu)
do
    /usr/local/cuda-11.4/bin/nvcc -o "${file%.cu}.o" ${file}
done
