// naive matrix multiplication
#include <iostream>

__global__ void mm_0(float* A, float* B, float* C, int N){
    //int threadId = threadIdx.x + blockDim.x * threadIdx.y
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    int sum = 0;
    for (int i = 0; i < N; i++){
        sum += A[row*N + i] * B[i*N + col];
    }
    C[row*N + col] = sum;
}