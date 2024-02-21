#ifndef MATRIX_MULTIPLICATION_KERNELS
#define MATRIX_MULTIPLICATION_KERNELS


void mm_cublas(thrust::device_vector<float> A, thrust::device_vector<float> B, thrust::device_vector<float> C, int N);
//__global__ void mm_0(float* A, float* B, float*C, int N);
//__global__ void mm_1(float* A, float* B, float*C, int N);

#endif