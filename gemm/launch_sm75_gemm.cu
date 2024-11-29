#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "utils.cuh"

using namespace cute;


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


int main(int argc, char** argv)
{
    int m = 10;
    int n = 10;
    int k = 10;

    using TA = half_t;
    using TB = half_t;
    using TC = float;

    cute::device_init(0);

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);
    thrust::host_vector<TC> h_C_cublas = generateRandomMatrix(m, n);




//     thrust::device_vector<TA> d_A = h_A;
//     thrust::device_vector<TB> d_B = h_B;
//     thrust::device_vector<TC> d_C = h_C;
//     thrust::device_vector<TC> d_C_cublas = h_C_cublas;
//
//     mm(d_A.data().get(), d_B.data().get(), d_C.data().get());
//     mm_cublas(d_A.data().get(), d_B.data().get(), d_C_cublas.data().get(), m, n, k);
//
//     h_C = d_C;
//     h_C_cublas = d_C_cublas;
//
//
//     //if (isSameMatrices(h_C.data(), h_C_cpu.data(), m, n) && isSameMatrices(h_C.data(), h_C_cublas.data(), m, n)) {
//     if (isSameMatrices(h_C.data(), h_C_cublas.data(), m, n)) {
//         printf("Correct answer\n");
//     } else {
//         printf("Wrong answer\n");
//     }


    return 0;
}