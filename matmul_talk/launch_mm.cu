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
    thrust::host_vector<float> hA = generateRandomMatrices(N);
    thrust::host_vector<float> hB = generateRandomMatrices(N);
    thrust::host_vector<float> hC(N*N);
    thrust::host_vector<float> hC_cublas(N*N);
    thrust::host_vector<float> bias(N);

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC = hC;
    thrust::device_vector<float> dC_cublas(N*N);
    thrust::device_vector<float> dbias = bias;

    //mm_naive(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()), thrust::raw_pointer_cast(dC.data()), N);
//    mm_global_memory_coalescing(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()), thrust::raw_pointer_cast(dC.data()), N);

//    mm_shared_memory_tiling(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()), thrust::raw_pointer_cast(dC.data()), N);
    mm_register_tiling(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()), thrust::raw_pointer_cast(dC.data()), N);
//    mm_vectorized_memory_access(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()), thrust::raw_pointer_cast(dC.data()), N);
//    mm_double_buffering(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()), thrust::raw_pointer_cast(dC.data()), N);
    mm_cublas(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dB.data()), thrust::raw_pointer_cast(dC.data()), N);



    #if 1
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