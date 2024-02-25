#ifndef MATRIX_MULTIPLICATION_KERNELS
#define MATRIX_MULTIPLICATION_KERNELS

__global__ void mm_0(float* A, float* B, float*C, int N);
__global__ void mm_1(float* A, float* B, float*C, int N);
__global__ void mm_2(float* A, float* B, float*C, int N);

#endif