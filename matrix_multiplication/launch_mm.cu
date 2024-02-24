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
    thrust::device_vector<float> dC_cublas(N*N);


    dim3 dimGrid(32, 32);
    dim3 dimBlock(32, 32);
    mm_0<<<dimGrid, dimBlock>>>(dA.data().get(), dB.data().get(), dC.data().get(), N);
    hC = dC;

    //
    // cublas
    //
    float alpha = 1.0f;
    float beta = 1.0f;

    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, thrust::raw_pointer_cast(dA.data()), N,
                     thrust::raw_pointer_cast(dB.data()), N, &beta, thrust::raw_pointer_cast(dC_cublas.data()), N);

    hC_cublas = dC_cublas;

    cublasDestroy(handle);

    if (isSameMatrices(hC.data().get(), hC_cublas.data().get(), N)==0){
        std::cout << "Wrong answer" << std::endl;
    }

    return 0;
}