#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "utils.cuh"
#include "sm75_gemm_test.cu"
#include "sm75_gemm_vectorized_load.cu"
#include "sm75_gemm_swizzle.cu"
#include "sm75_gemm_ldsm.cu"
#include "sm75_gemm_smem_buffering.cu"



#include "sm75_gemm_vectorized_load_256.cu"
#include "sm75_gemm_vectorized_gmem_store_256.cu"
#include "sm75_gemm_swizzle_256.cu"
#include "sm75_gemm_ldsm_256.cu"
#include "sm75_gemm_smem_pipelining_256.cu"
#include "sm75_gemm_register_pipelining_256.cu"



using namespace cute;


int main(int argc, char** argv)
{
    int m = 128;
    if (argc >= 2)
    sscanf(argv[1], "%d", &m);

    int n = 128;
    if (argc >= 3)
    sscanf(argv[2], "%d", &n);

    int k = 32;
    if (argc >= 4)
    sscanf(argv[3], "%d", &k);

    using TA = half_t;
    using TB = half_t;
    using TC = float;

    cute::device_init(0);

    thrust::host_vector<TA> h_A = generateRandomMatrix<TA> (m * k);
    thrust::host_vector<TB> h_B = generateRandomMatrix<TB> (n * k);
    thrust::host_vector<TC> h_C(m * n, 0.0f);
    thrust::host_vector<TC> h_C_cublas(m * n, 0.0f);


    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;
    thrust::device_vector<TC> d_C_cublas = h_C_cublas;

    //gemm_test(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);


    gemm_cublas(d_A.data().get(), d_B.data().get(), d_C_cublas.data().get(), m, n, k);
    h_C_cublas = d_C_cublas;
//
//
    //if (isSameMatrices(h_C.data(), h_C_cpu.data(), m, n) && isSameMatrices(h_C.data(), h_C_cublas.data(), m, n)) {
//     gemm_vectorized_load(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
//     h_C = d_C;
//     isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "vectorized_load");
//
//     gemm_swizzle(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
//     h_C = d_C;
//     isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "swizzle");
//
//     gemm_ldsm(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
//     h_C = d_C;
//     isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "ldsm");
// //
//     gemm_smem_buffering(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
//     h_C = d_C;
//     isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "smem_buffering");
//
//     gemm_test(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
//     h_C = d_C;
//     isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "test");
//
    gemm_vectorized_load_256(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
    h_C = d_C;
    isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "vectorized_load_256");

    gemm_vectorized_gmem_store_256(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
    h_C = d_C;
    isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "vectorized_gmem_store_256");

//     for (int i=0;i<32;i++) {
//         printf("cutlass = %f, cublas = %f\n", h_C[i], h_C_cublas[i]);
//     }

    gemm_swizzle_256(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
    h_C = d_C;
    isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "swizzle_256");

//     for (int i=0;i<32;i++) {
//         printf("cutlass = %f, cublas = %f\n", h_C[i], h_C_cublas[i]);
//     }

    gemm_ldsm_256(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
    h_C = d_C;
    isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "ldsm_256");

    gemm_smem_pipelining_256(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
    h_C = d_C;
    isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "smem_pipelining_256");

    gemm_register_pipelining_256(d_A.data().get(), d_B.data().get(), d_C.data().get(), m, n, k);
    h_C = d_C;
    isSameMatrices(h_C.data(), h_C_cublas.data(), m * n, "smem_register_256");



    return 0;
}