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
    int N = 4096;
    thrust::host_vector<float> hA = generateTestMatrices(N);
    thrust::host_vector<float> hB = generateTestMatrices(N);
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
        std::cout << "Running kernel 2" << std::endl;
        int TILE_WIDTH = 32;
        dim3 gridDim_mm_new_2(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_2(TILE_WIDTH,TILE_WIDTH);
        mm_new_2<<<gridDim_mm_new_2, blockDim_mm_new_2>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif

    #if 0
    {
//        for (int i=128;i<256; i++){
//                printf("%d %f\n", i, hA[i]);
//         }

//        for (int i=0;i<128; i++){
//            printf("%d %f\n", i, hA[i]);
//        }
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_3(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_3(256);
        std::cout << "Running kernel 3" << std::endl;
        mm_new_3<<<gridDim_mm_new_3, blockDim_mm_new_3>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif

    #if 0
    {
//        for (int i=128;i<256; i++){
//                printf("%d %f\n", i, hA[i]);
//         }

//        for (int i=0;i<128; i++){
//            printf("%d %f\n", i, hA[i]);
//        }
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_4(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_4(256);
        std::cout << "Running kernel 4" << std::endl;
        mm_new_4<<<gridDim_mm_new_4, blockDim_mm_new_4>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif


    #if 0
    {
//        for (int i=128;i<256; i++){
//                printf("%d %f\n", i, hA[i]);
//         }

//        for (int i=0;i<128; i++){
//            printf("%d %f\n", i, hA[i]);
//        }
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_5(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_5(256);
        std::cout << "Running kernel 5" << std::endl;
        mm_new_5<<<gridDim_mm_new_5, blockDim_mm_new_5>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif

    #if 0
    {
//        for (int i=128;i<256; i++){
//                printf("%d %f\n", i, hA[i]);
//         }

//        for (int i=0;i<128; i++){
//            printf("%d %f\n", i, hA[i]);
//        }
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_6(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_6(256);
        std::cout << "Running kernel 6" << std::endl;
        mm_new_6<<<gridDim_mm_new_6, blockDim_mm_new_6>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif


    #if 0
    {
//        for (int i=128;i<256; i++){
//                printf("%d %f\n", i, hA[i]);
//         }

//        for (int i=0;i<128; i++){
//            printf("%d %f\n", i, hA[i]);
//        }
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_7(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_7(256);
        std::cout << "Running kernel 7" << std::endl;
        mm_new_7<<<gridDim_mm_new_7, blockDim_mm_new_7>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif


    #if 0
    {
//        for (int i=128;i<256; i++){
//                printf("%d %f\n", i, hA[i]);
//         }

//        for (int i=0;i<128; i++){
//            printf("%d %f\n", i, hA[i]);
//        }
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_8(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_8(256);
        std::cout << "Running kernel 8" << std::endl;
        mm_new_8<<<gridDim_mm_new_8, blockDim_mm_new_8>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif

    #if 0
    {
//        for (int i=128;i<256; i++){
//                printf("%d %f\n", i, hA[i]);
//         }

//        for (int i=0;i<128; i++){
//            printf("%d %f\n", i, hA[i]);
//        }
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_8_copy(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_8_copy(256);
        std::cout << "Running kernel 8 copy" << std::endl;
        mm_new_8_copy<<<gridDim_mm_new_8_copy, blockDim_mm_new_8_copy>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif


    #if 0
    {
//        for (int i=128;i<256; i++){
//                printf("%d %f\n", i, hA[i]);
//         }

//        for (int i=0;i<128; i++){
//            printf("%d %f\n", i, hA[i]);
//        }
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_8_float4(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_8_float4(256);
        std::cout << "Running kernel 8 float4" << std::endl;
        mm_new_8_float4<<<gridDim_mm_new_8_float4, blockDim_mm_new_8_float4>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif




    #if 0
    {
//        for (int i=128;i<256; i++){
//                printf("%d %f\n", i, hA[i]);
//         }

//        for (int i=0;i<128; i++){
//            printf("%d %f\n", i, hA[i]);
//        }
        int TILE_WIDTH = 128;
        dim3 gridDim_mm_new_9(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_mm_new_9(256);
        std::cout << "Running kernel 9" << std::endl;
        mm_new_9<<<gridDim_mm_new_9, blockDim_mm_new_9>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif


    #if 0
    {
        int TILE_WIDTH = 128;
        dim3 blockDim_yz(256);
        dim3 gridDim_yz(N / TILE_WIDTH,N / TILE_WIDTH);
        mysgemm_v9<<<gridDim_yz, blockDim_yz>>>(N,N,N,1.0f,thrust::raw_pointer_cast(dA.data()),thrust::raw_pointer_cast(dB.data()),0.0f,thrust::raw_pointer_cast(dC.data()));
        mysgemm_v11<<<gridDim_yz, blockDim_yz>>>(N,N,N,1.0f,thrust::raw_pointer_cast(dA.data()),thrust::raw_pointer_cast(dB.data()),0.0f,thrust::raw_pointer_cast(dC.data()));

        hC = dC;
    }
    #endif



    #if 0
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



    #if 0
    {
        //
        // cublas column major
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
        //
        //
        //
    }
    #endif


//llm.c

    #if 1
    {
        int TILE_WIDTH = 128;
        dim3 gridDim_llmc_1(N / TILE_WIDTH,N / TILE_WIDTH);
        dim3 blockDim_llmc_1(256);
        std::cout << "Running llmc kernel 1" << std::endl;
        mm_new_8_float4<<<gridDim_llmc_1, blockDim_llmc_1>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                               thrust::raw_pointer_cast(dC.data()), N);
        hC = dC;
    }
    #endif


    #if 0
    {
        // A row major
        // B column major
        // C column major
        float alpha = 1.0f;
        float beta = 1.0f;

        cudaError_t cudaStat;  // cudaMalloc status
        cublasStatus_t stat;   // cuBLAS functions status
        cublasHandle_t handle; // cuBLAS context
        cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, thrust::raw_pointer_cast(dA.data()), N,
                         thrust::raw_pointer_cast(dB.data()), N, &beta, thrust::raw_pointer_cast(dC_cublas.data()), N);

        hC_cublas = dC_cublas;

        cublasDestroy(handle);
        //
        //
        //
    }
    #endif





    #if 0
        if (isSameMatrices(hC.data(), hC_cublas.data(), N)==0){
//        for (int i=0;i<N;i += 128){
//            for (int j=0;j<N; j+=128){
//                std::cout << N * i + j << " " << hC[N * i + j] << " " << hC_cublas[N * i + j] << std::endl;
//            }
//        }

//            for (int i=0;i<100;i++){
//                std::cout << i << " " << hC[i] << " " << hC_cublas[i] << std::endl;
//            }
        for (int i=0;i<32;i++){
            for (int j=0;j<32;j++){
                std::cout << hC[i + N*j] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\n";
        std::cout << "\n";
        std::cout << "\n";
        std::cout << "\n";
        std::cout << "\n";

        for (int i=0;i<32;i++){
            for (int j=0;j<32;j++){
                std::cout << hC_cublas[i + N*j] << " ";
            }
            std::cout << "\n";
        }
        //        int num = countZeros(hC.data(), N);
        //        std::cout << "number of zeros in hC is " << num << std::endl;
            std::cout << "Wrong answer" << std::endl;
        } else {
            std::cout << "Correct answer" << std::endl;
        }
    #endif


    return 0;
}