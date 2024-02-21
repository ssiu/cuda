//https://github.com/siboehm/SGEMM_CUDA/blob/master/cuBLAS_sgemm.cu
#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



void mm_cublas(thrust::device_vector<float> A, thrust::device_vector<float> B, thrust::device_vector<float> C, int N) {
    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context

    float alpha = 1.0f;
    float beta = 1.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N,
                     d_B, N, &beta, d_C, N);

}