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
    int N = 2048;

    thrust::host_vector<float> hA = generateMatrices(N);
    thrust::host_vector<float> hB = generateMatrices(N);
    thrust::host_vector<float> hC(N*N);
    thrust::host_vector<float> hC_cublas(N*N);

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC = hC;
    thrust::device_vector<float> dC_cublas(N*N);


//    dim3 dimGrid(32, 32);
//    dim3 dimBlock(32, 32);
//    mm_0<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                    thrust::raw_pointer_cast(dC.data()), N);
//
//    mm_1<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                    thrust::raw_pointer_cast(dC.data()), N);
//    mm_2<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                    thrust::raw_pointer_cast(dC.data()), N);
//
//
//    dim3 dimGrid3(64, 64);
//    dim3 dimBlock3(8, 32);
//    mm_3<<<dimGrid3, dimBlock3>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
//                                    thrust::raw_pointer_cast(dC.data()), N);




    //
    // cublas
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
    //
    //
    //

    dim3 dimGrid4(16, 16);
    dim3 dimBlock4(256, 1);
    mm_4<<<dimGrid4, dimBlock4>>>(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()),
                                   thrust::raw_pointer_cast(dC.data()), N);

    hC = dC;


    if (isSameMatrices(hC.data(), hC_cublas.data(), N)==0){
        for (int i=0;i<2048;i += 128){
            for (int j=0;j<2048; j+=128){
                std::cout << 2048 * i + j << " " << hC[2048 * i + j] << " " << hC_cublas[2048 * i + j] << std::endl;
            }
        }

//        for (int i=0;i<100;i++){
//            std::cout << hC[i] << " " << hC_cublas[i] << std::endl;
//        }
        num = countZeros(hC);
        std::cout << "number of zeros in hC is "<< num << std::endl;
        std::cout << "Wrong answer" << std::endl;
    } else {
        std::cout << "Correct answer" << std::endl;
    }

    return 0;
}