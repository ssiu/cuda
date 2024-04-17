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

//uncomment mm4,mm7,mm9

//    dim3 dimGrid(64, 64);
//    dim3 dimBlock(32, 32);
////    mm_0<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
////                                    thrust::raw_pointer_cast(dC.data()), N);
////
//    mm_1<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                    thrust::raw_pointer_cast(dC.data()), N);
//    mm_2<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                    thrust::raw_pointer_cast(dC.data()), N);
////
////
//    dim3 dimGrid3(64, 64);
//    dim3 dimBlock3(8, 32);
//    mm_3<<<dimGrid3, dimBlock3>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                    thrust::raw_pointer_cast(dC.data()), N);

//    dim3 dimGrid4(N / TILE_WIDTH, N / TILE_WIDTH);
//    dim3 dimBlock4(256, 1);
//    mm_4<<<dimGrid4, dimBlock4>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                   thrust::raw_pointer_cast(dC.data()), N);

////////////////////////////////////////////////////////////////////////////////////////////////
//    mm_5<<<dimGrid4, dimBlock4>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                   thrust::raw_pointer_cast(dC.data()), N);
//
////// Device code
////__global__ void MyKernel(...)
////{
////    extern __shared__ float buffer[];
////    ...
////}
////
////// Host code
//    int maxbytes = 98304; // 96 KB
//    cudaFuncSetAttribute(mm_6, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
//    mm_6<<<dimGrid4, dimBlock4>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                   thrust::raw_pointer_cast(dC.data()), N);
////////////////////////////////////////////////////////////////////////////////////////////////////
//    mm_7<<<dimGrid4, dimBlock4>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                               thrust::raw_pointer_cast(dC.data()), N);

//    mm_8<<<dimGrid4, dimBlock4>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                               thrust::raw_pointer_cast(dC.data()), N);
//    mm_9<<<dimGrid4, dimBlock4>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                               thrust::raw_pointer_cast(dC.data()), N);
//    hC = dC;


    // yz
//    dim3 blockDim(256);
//    dim3 gridDim(N / TILE_WIDTH,N / TILE_WIDTH);
//    mysgemm_v9<<<gridDim, blockDim>>>(N,N,N,1.0f,thrust::raw_pointer_cast(dA.data()),thrust::raw_pointer_cast(dB.data()),0.0f,thrust::raw_pointer_cast(dC.data()));
//    mysgemm_v11<<<gridDim, blockDim>>>(N,N,N,1.0f,thrust::raw_pointer_cast(dA.data()),thrust::raw_pointer_cast(dB.data()),0.0f,thrust::raw_pointer_cast(dC.data()));
//
//    hC = dC;

    #if 0
    {
        int TILE_WIDTH = 32;
        dim3 gridDim_mm_new_1(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_1(TILE_WIDTH,TILE_WIDTH);
        mm_new_1<<<gridDim_mm_new_1, blockDim_mm_new_1>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif

    #if 0
    {
        int TILE_WIDTH = 32;
        dim3 gridDim_mm_new_2(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_2(TILE_WIDTH,TILE_WIDTH);
        mm_new_2<<<gridDim_mm_new_2, blockDim_mm_new_2>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif

    #if 1
    {
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_3(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_3(TILE_WIDTH,TILE_WIDTH);
        mm_new_3<<<gridDim_mm_new_3, blockDim_mm_new_3>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif

    #if 1
    {
        //
        // cublas row major
        //
        float alpha = 1.0f;
        float beta = 1.0f;

        cudaError_t cudaStat;  // cudaMalloc status
        cublasStatus_t stat;   // cuBLAS functions status
        cublasHandle_t handle; // cuBLAS context
        cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, thrust::raw_pointer_cast(dB.data()), N,
                         thrust::raw_pointer_cast(dA.data()), N, &beta, thrust::raw_pointer_cast(dC_cublas.data()), N);

        hC_cublas = dC_cublas;

        cublasDestroy(handle);
    }
    #endif






    //
    // cublas column major
    //
//    float alpha = 1.0f;
//    float beta = 1.0f;
//
//    cudaError_t cudaStat;  // cudaMalloc status
//    cublasStatus_t stat;   // cuBLAS functions status
//    cublasHandle_t handle; // cuBLAS context
//    cublasCreate(&handle);
//    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, thrust::raw_pointer_cast(dA.data()), N,
//                     thrust::raw_pointer_cast(dB.data()), N, &beta, thrust::raw_pointer_cast(dC_cublas.data()), N);
//
//    hC_cublas = dC_cublas;
//
//    cublasDestroy(handle);
//    //
//    //
//    //



    if (isSameMatrices(hC.data(), hC_cublas.data(), N)==0){
//        for (int i=0;i<N;i += 128){
//            for (int j=0;j<N; j+=128){
//                std::cout << N * i + j << " " << hC[N * i + j] << " " << hC_cublas[N * i + j] << std::endl;
//            }
//        }

        for (int i=0;i<100;i++){
            std::cout << i << " " << hC[i] << " " << hC_cublas[i] << std::endl;
        }
//        int num = countZeros(hC.data(), N);
//        std::cout << "number of zeros in hC is " << num << std::endl;
        std::cout << "Wrong answer" << std::endl;
    } else {
        std::cout << "Correct answer" << std::endl;
    }

    return 0;
}