#ifndef MATRIX_MULTIPLICATION_KERNELS
#define MATRIX_MULTIPLICATION_KERNELS

void mm_naive(float* A, float* B, float* C, int N);
void mm_cublas(float* A, float* B, float* C, int N);
void mm_global_memory_coalescing(float* A, float* B, float* C, int N);
void mm_shared_memory_tiling(float* A, float* B, float* C, int N)
void mm_double_buffering(float* A, float* B, float* C, int N);

#endif