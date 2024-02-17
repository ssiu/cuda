#include <iostream>
#include <string>

// This kernel
// variables: number of threads
//            number of ILP

#define N 1048576

__global__ void arithmetic_kernel_1() {
    int a = 1;

    #pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}


__global__ void arithmetic_kernel_2() {
    int a = 1;

    #pragma unroll 2
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}

__global__ void arithmetic_kernel_3() {
    int a = 1;

    #pragma unroll 3
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}

__global__ void arithmetic_kernel_4() {
    int a = 1;

    #pragma unroll 4
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}

__global__ void arithmetic_kernel_5() {
    int a = 1;

    #pragma unroll 5
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}

int main(){

    arithmetic_kernel_1<<<1024, 128>>>();
    arithmetic_kernel_2<<<1024, 128>>>();
    arithmetic_kernel_3<<<1024, 128>>>();
    arithmetic_kernel_4<<<1024, 128>>>();
    arithmetic_kernel_5<<<1024, 128>>>();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }
    return 0;

}