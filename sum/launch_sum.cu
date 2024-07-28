#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>
#include "sum.cuh"
#include "utils.cuh"



int main(){
    int N = static_cast<int>(10);

    thrust::host_vector<float> h_in = generateRandomArray(N);
    thrust::host_vector<float> h_out(1);
    thrust::host_vector<float> h_out_cub(1);

    thrust::device_vector<float> d_in = h_in;

    thrust::device_vector<float> d_out(1);
    thrust::device_vector<float> d_out_cub(1);

    sum_cub(thrust::raw_pointer_cast(dA.data()), thrust::raw_pointer_cast(dA.data(d_out_cub[0])), N);


    h_out_cub[0] = d_out_cub[0];
    std::cout << h_out_cub[0] << std::endl;

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