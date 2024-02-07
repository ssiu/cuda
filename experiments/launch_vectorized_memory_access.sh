#!/bin/bash


# Start with 1024 (2^10)
number=$((2**10))

# Set a maximum limit if necessary (e.g., less than 2^20)
max_limit=$((2**12))

while [ $number -le $max_limit ]; do
  echo Currently profiling array of size $number
  nvprof ./vectorized_memory_access.o $number
  #nvprof --csv --log-file "profile_$number.csv" ./vectorized_memory_access.cu $number
  number=$((number * 2))
done


