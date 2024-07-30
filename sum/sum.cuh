#ifndef SUM_KERNELS
#define SUM_KERNELS


void sum_naive(float* d_in, float* d_out, int N);
void sum_cub(float* d_in, float* d_out, int N);

#endif