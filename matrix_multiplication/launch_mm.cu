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
//    int N = 1024;
//
//    thrust::host_vector<float> hA = generateMatrices(N);
//    thrust::host_vector<float> hB = generateMatrices(N);
//    thrust::host_vector<float> hC(N*N);
//    thrust::host_vector<float> hC_cublas(N*N);
//
//    thrust::device_vector<float> dA = hA;
//    thrust::device_vector<float> dB = hB;
//    thrust::device_vector<float> dC = hC;
//    thrust::device_vector<float> dC_cublas(N*N);
//
//
//    dim3 dimGrid(32, 32);
//    dim3 dimBlock(32, 32);
//    mm_0<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                    thrust::raw_pointer_cast(dC.data()), N);
//    hC = dC;
//
//    //
//    // cublas
//    //
//    float alpha = 1.0f;
//    float beta = 1.0f;
//
//    cudaError_t cudaStat;  // cudaMalloc status
//    cublasStatus_t stat;   // cuBLAS functions status
//    cublasHandle_t handle; // cuBLAS context
//    cublasCreate(&handle);
//    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &alpha, thrust::raw_pointer_cast(dB.data()), N,
//                     thrust::raw_pointer_cast(dA.data()), N, &beta, thrust::raw_pointer_cast(dC_cublas.data()), N);
//
//    hC_cublas = dC_cublas;
//
//    cublasDestroy(handle);
//
//    for (int i=0;i<100;i++){
//        std::cout << "check A and B matrices: " << hA[i] << " " << hB[i] << std::endl;
//    }
//
//    for (int i=0;i<100;i++){
//        std::cout << "compare results against cublas: " << hC[i] << " " << hC_cublas[i] << std::endl;
//    }
//
//    if (isSameMatrices(hC.data(), hC_cublas.data(), N)==0){
//        std::cout << "Wrong answer" << std::endl;
//    }

// cublas experiment
    thrust::host_vector<float> hA(4);
    thrust::host_vector<float> hB(4);
    thrust::host_vector<float> hC_cublas(4);

    hA[0] = static_cast<float>(0.0);
    hA[1] = static_cast<float>(2.0);
    hA[2] = static_cast<float>(1.0);
    hA[3] = static_cast<float>(3.0);
    hB[0] = static_cast<float>(4.0);
    hB[1] = static_cast<float>(6.0);
    hB[2] = static_cast<float>(5.0);
    hB[3] = static_cast<float>(7.0);
//    for (int i=0;i<4;i++){
//        hA[i] = static_cast<float>(i);
//        hB[i] = static_cast<float>(i+4);
//    }

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC_cublas(4);


    float alpha = 1.0f;
    float beta = 0.0f;

    cudaError_t cudaStat;  // cudaMalloc status
    cublasStatus_t stat;   // cuBLAS functions status
    cublasHandle_t handle; // cuBLAS context
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &alpha, thrust::raw_pointer_cast(dA.data()), 2,
                     thrust::raw_pointer_cast(dB.data()), 2, &beta, thrust::raw_pointer_cast(dC_cublas.data()), 2);

//    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 2, 2, 2, &alpha, thrust::raw_pointer_cast(dA.data()), 2,
//                     thrust::raw_pointer_cast(dB.data()), 2, &beta, thrust::raw_pointer_cast(dC_cublas.data()), 2);


    hC_cublas = dC_cublas;

    cublasDestroy(handle);
    for (int i=0;i<4;i++){
        std::cout << "A: " << hA[i] << std::endl;
    }
    for (int i=0;i<4;i++){
        std::cout << "B: " << hB[i] << std::endl;
    }
    for (int i=0;i<4;i++){
        std::cout << "cublas: " << hC_cublas[i] << std::endl;
    }

    return 0;
}