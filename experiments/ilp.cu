#include <iostream>
#include <string>

// This kernel
// variables: number of threads
//            number of ILP

// #define N 10000000

__global__ void arithmetic_kernel_1(int N) {
    int a = 1;

    #pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}


__global__ void arithmetic_kernel_2(int N) {
    int a = 1;

    #pragma unroll 2
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}

__global__ void arithmetic_kernel_3(int N) {
    int a = 1;

    #pragma unroll 3
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}

__global__ void arithmetic_kernel_4(N) {
    int a = 1;

    #pragma unroll 4
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}

__global__ void arithmetic_kernel_5(N) {
    int a = 1;

    #pragma unroll 5
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}

int main(int argc, char *argv[]){
    int N = std::stoi(argv[1]);

    arithmetic_kernel_1<<<1024, 128>>>(N);
    arithmetic_kernel_2<<<1024, 128>>>(N);
    arithmetic_kernel_3<<<1024, 128>>>(N);
    arithmetic_kernel_4<<<1024, 128>>>(N);
    arithmetic_kernel_5<<<1024, 128>>>(N);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }
    return 0;

}