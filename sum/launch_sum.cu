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
    int N = static_cast<int>(1<<20);

    thrust::host_vector<float> h_in = generateRandomArray(N);
    thrust::host_vector<float> h_out(1);
    thrust::host_vector<float> h_out_cub(1);

    thrust::device_vector<float> d_in = h_in;

    thrust::device_vector<float> d_out(1);
    thrust::device_vector<float> d_out_cub(1);


    sum_naive(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), N);
    h_out[0] = d_out[0];
    std::cout << h_out[0] << std::endl;

    sum_vectorized(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), N);
    h_out[0] = d_out[0];
    std::cout << h_out[0] << std::endl;

    sum_cub(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out_cub.data()), N);
    h_out_cub[0] = d_out_cub[0];
    std::cout << h_out_cub[0] << std::endl;



    return 0;
}