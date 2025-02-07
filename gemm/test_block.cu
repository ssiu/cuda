#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>


__global__ void my_test_kernel() {
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        printf("gridDim.z = %d\n", gridDim.z);
    }
}


void my_test() {
    dim3 dimBlock(1); // Example block size
    dim3 dimGrid(1, 1, 2);    // Only using Z-dimension

    my_kernel<<<dimGrid, dimBlock>>>();

}