#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>


__global__ void my_test_kernel() {

    printf("blockIdx.z = %d, gridIdx.z = %d\n", blockIdx.z, gridIdx.z);

}


void my_test() {
    dim3 dimBlock(1); // Example block size
    dim3 dimGrid(1, 1, 2);    // Only using Z-dimension

    my_test_kernel<<<dimGrid, dimBlock>>>();

}