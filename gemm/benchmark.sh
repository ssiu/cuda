#!/bin/bash

# Array of values to loop through
sizes=(512 1024 2048 4096 8192)

for size in "${sizes[@]}"; do
    echo "Running with size $size..."
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --csv ./launch_sm75_gemm.o "$size" "$size" "$size" > "${size}.csv"

    echo "Output saved to ${size}.csv"

done

python plot_kernels.py

rm -rf *.csv

ncu -f --target-processes all --set full \
--import-source on \
-o sm75_gemm_profile ./launch_sm75_gemm.o 8192 8192 8192