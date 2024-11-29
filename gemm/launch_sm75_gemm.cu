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


int main(int argc, char** argv)
{
    int m = 128;
    int n = 128;
    int k = 128;

    using TA = half_t;
    using TB = half_t;
    using TC = float;

    cute::device_init(0);

    thrust::host_vector<TA> h_A(m*k) = generateRandomMatrix<TA> (m * k);
    thrust::host_vector<TB> h_B(n*k) = generateRandomMatrix<TB> (n * k);
    thrust::host_vector<TC> h_C(m*n);
    thrust::host_vector<TC> h_C_cublas = generateRandomMatrix<TC>(m * n);

    for (int i=0; i< 10; i++) {
        for (int j=0; j< 10; j++) {

            printf("%f ", h_C_cublas[i*10+j]);

        }
        printf("\n");

    }





    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    thrust::device_vector<TC> d_C_cublas = h_C_cublas;

    mm_cublas(d_A.data().get(), d_B.data().get(), d_C_cublas.data().get(), M, N, K);
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