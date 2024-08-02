#ifndef SUM_KERNELS
#define SUM_KERNELS


void sum_naive(float* d_in, float* d_out, int N);
void sum_vectorized(float* d_in, float* d_out, int N);
void sum_atomic_add(float* d_in, float* d_out, int N);
void sum_warp_shuffle(float* d_in, float* d_out, int N);
void sum_cub(float* d_in, float* d_out, int N);

#endif