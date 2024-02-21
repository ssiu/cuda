//https://github.com/siboehm/SGEMM_CUDA/blob/master/cuBLAS_sgemm.cu
#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



void mm_cublas(thrust::device_vector<float> dA, thrust::device_vector<float> dB, thrust::device_vector<float> dC, int N) {
    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context

    float alpha = 1.0f;
    float beta = 1.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA, N,
                     dB, N, &beta, dC, N);

}