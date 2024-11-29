#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include <cmath> // For std::fabs
#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// todo: investigate the correct epsilon
bool areFloatsEqual(float a, float b, float epsilon = 1e-5f) {
    return std::fabs(a - b) < epsilon;
}


int isSameMatrices(float* A_1, float* A_2, int M, int N){

    for (int i = 0; i < M*N; i++){
        if (!(areFloatsEqual(A_1[i], A_2[i]))) {
            //std::cout << "Wrong answer:" << A_1[i] << " " << A_2[i] << std::endl;
            return 0;
        }
    }
    return 1;
}

template <typename T>
thrust::host_vector<T> generateRandomMatrix(int M, int N) {
    thrust::host_vector<T> A(M * N);

    for (int i = 0; i < M * N; i++) {
        A[i] = static_cast<T>( 2*(rand() / double(RAND_MAX)) - 1 );
    }

    // Return both matrices
    return A;
}

void mm_cublas(half_t* A, half_t* B, float* C,
                int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    cublasCreate(&handle);
    //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 16, 8, 8, &alpha, A, 16, B, 8, &beta, C, 16);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                            A, CUDA_R_16F, M,
                            B, CUDA_R_16F, K, &beta,
                            C, CUDA_R_32F, M,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);
}
