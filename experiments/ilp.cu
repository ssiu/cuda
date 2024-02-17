#include <iostream>
#include <string>

// This kernel
// variables: number of threads
//            number of ILP

// #define N 10000000

__global__ void arithmetic_kernel_1(int N) {
    int a = 1;

    //#pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}


__global__ void arithmetic_kernel_2(int N) {
    int a = 1;
    int b = 1;

    //#pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
        b = b * 1 + 1;
    }
}

__global__ void arithmetic_kernel_3(int N) {
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

__global__ void arithmetic_kernel_4(int N) {
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

__global__ void arithmetic_kernel_5(int N) {
    int a = 1;
    int b = 1;
    int c = 1;
    int d = 1;
    int e = 1;

    //#pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
        b = b * 1 + 1;
        c = c * 1 + 1;
        d = d * 1 + 1;
        e = e * 1 + 1;
    }
}

int main(int argc, char *argv[]){

    int N = std::stoi(argv[1]);

    int numBlocks = 1024;
    int numThreads = 2048;
    arithmetic_kernel_1<<<numBlocks, numThreads>>>(N);
    arithmetic_kernel_2<<<numBlocks, numThreads>>>(N);
    arithmetic_kernel_3<<<numBlocks, numThreads>>>(N);
    arithmetic_kernel_4<<<numBlocks, numThreads>>>(N);
    arithmetic_kernel_5<<<numBlocks, numThreads>>>(N);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }
    return 0;

}