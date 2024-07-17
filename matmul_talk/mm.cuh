#ifndef MATRIX_MULTIPLICATION_KERNELS
#define MATRIX_MULTIPLICATION_KERNELS

void mm_naive(float* A, float* B, float* C, int N);
void mm_cublas(float* A, float* B, float* C, int N);

#endif