// naive matrix multiplication
#include <iostream>


__global__ void mm_naive_kernel(float* A, float* B, float* C, int N){
    //int threadId = threadIdx.x + blockDim.x * threadIdx.y
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    float sum = 0;
    for (int i = 0; i < N; i++){
        sum += A[row*N + i] * B[i*N + col];
//        if (blockIdx.x == 0 and blockIdx.x==0 and threadIdx.x==0 and threadIdx.y==0){
//            printf("A is %f, B is %f, SUM is %f\n", A[row*N + i], B[i*N + col], sum);
//        }
    }
    C[row*N + col] = sum;
}

void mm_naive(float* A, float* B, float* C, int N) {
    dim3 dimGrid(N / 32, N / 32);
    dim3 dimBlock(32, 32);
    mm_naive_kernel<<<dimGrid, dimBlock>>>(A, B, C, N);
}