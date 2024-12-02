#!/bin/bash

output_csv="compute_throughput.csv"
echo "Size,Compute_Throughput" > $output_csv

# Array of values to loop through
sizes=(512 1024 2048 4096 8192)

for size in "${sizes[@]}"; do
    echo "Running with size $size..."
    throughput=$(ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed_time \
                     --set full \
                     --csv ./launch_sm75_gemm.o $size $size\
                     | grep "sm__throughput.avg.pct_of_peak_sustained_elapsed_time" \
                     | awk -F ',' '{print $2}')
    echo "$size,$throughput" >> $output_csv
done