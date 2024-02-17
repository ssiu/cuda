#include <iostream>
#include <string>

// This kernel
// variables: number of threads
//            number of ILP

#define N 1000000000

__global__ void arithmetic_kernel_1() {
    int a = 1;

    //#pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}


__global__ void arithmetic_kernel_2() {
    int a = 1;
    int b = 1;

    //#pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
        b = b * 1 + 1;
    }
}

__global__ void arithmetic_kernel_3() {
    int a = 1;
    int b = 1;
    int c = 1;

    //#pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
        b = b * 1 + 1;
        c = c * 1 + 1;
    }
}

__global__ void arithmetic_kernel_4() {
    int a = 1;
    int b = 1;
    int c = 1;
    int d = 1;

    //#pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
        b = b * 1 + 1;
        c = c * 1 + 1;
        d = d * 1 + 1;
    }
}

__global__ void arithmetic_kernel_5() {
    int a = 1;
    int b = 1;
    int c = 1;
    int d = 1;
    int e = 1;

    #pragma unroll
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
        b = b * 1 + 1;
        c = c * 1 + 1;
        d = d * 1 + 1;
        e = e * 1 + 1;
    }
}

int main(){

    int numBlocks = 1024;
    int numThreads = 256;
    arithmetic_kernel_1<<<numBlocks, numThreads>>>();
    arithmetic_kernel_2<<<numBlocks, numThreads>>>();
    arithmetic_kernel_3<<<numBlocks, numThreads>>>();
    arithmetic_kernel_4<<<numBlocks, numThreads>>>();
    arithmetic_kernel_5<<<numBlocks, numThreads>>>();

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }
    return 0;

}