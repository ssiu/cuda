#!/bin/bash

for file in $(ls | grep _mm.o$)
do
    echo /usr/local/cuda-11.4/bin/ncu -f --set full -o "profile_${file}" ./${file}
    /usr/local/cuda-11.4/bin/ncu -f --set full -o "profile_${file}" ./${file}
done