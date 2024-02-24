#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include "mm.cuh"
#include "utils.cuh"


int main(){
    int N = 1024;

    thrust::host_vector<float> hA = generateMatrices(N);
    thrust::host_vector<float> hB = generateMatrices(N);
    thrust::host_vector<float> hC(N*N);
    thrust::host_vector<float> hC_cublas(N*N);

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC = hC;
    thrust::device_vector<float> dC_cublas = hC_cublas;


    dim3 dimGrid(32, 32);
    dim3 dimBlock(32, 32);
    //mm_0<<<dimGrid, dimBlock>>>(dA.data().get(), dB.data().get(), dC.data().get(), N);

    //
    // cublas
    //
    float alpha = 1.0f;
    float beta = 1.0f;

    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dA.data().get(), N,
                     dB.data().get(), N, &beta, dC_cublas.data().get(), N);

    hC_cublas = dC_cublas;

    return 0;
}