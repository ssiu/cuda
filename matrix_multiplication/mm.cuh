#ifndef MATRIX_MULTIPLICATION_KERNELS
#define MATRIX_MULTIPLICATION_KERNELS

__global__ void mm_0(float* A, float* B, float*C, int N);
__global__ void mm_1(float* A, float* B, float*C, int N);
__global__ void mm_2(float* A, float* B, float*C, int N);
__global__ void mm_3(float* A, float* B, float*C, int N);
__global__ void mm_4(float* A, float* B, float*C, int N);
__global__ void mm_5(float* A, float* B, float*C, int N);
__global__ void mm_6(float* A, float* B, float*C, int N);
__global__ void mm_7(float* A, float* B, float*C, int N);
__global__ void mm_8(float* A, float* B, float*C, int N);
__global__ void mm_9(float* A, float* B, float*C, int N);
__global__ void mm_new_1(float* A, float* B, float*C, int N);
__global__ void mm_new_2(float* A, float* B, float*C, int N);
__global__ void mm_new_3(float* A, float* B, float*C, int N);
__global__ void mm_new_4(float* A, float* B, float*C, int N);
__global__ void mm_new_5(float* A, float* B, float*C, int N);
__global__ void mm_new_6(float* A, float* B, float*C, int N);
__global__ void mm_new_7(float* A, float* B, float*C, int N);
__global__ void mm_new_8(float* A, float* B, float*C, int N);
__global__ void mm_new_8_float4(float* A, float* B, float*C, int N);
__global__ void mm_new_9(float* A, float* B, float*C, int N);
__global__ void mysgemm_v9(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
__global__ void mysgemm_v11(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C);
#endif