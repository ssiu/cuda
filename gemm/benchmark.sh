#!/bin/bash

# Array of values to loop through
sizes=(512 1024 2048 4096 8192)

for size in "${sizes[@]}"; do
    echo "Running with size $size..."
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --csv ./launch_sm75_gemm.o $size $size $size > $(size).csv

done