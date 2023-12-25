#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/mma.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;


__global__ void mma_atom(half_t* dA, half_t* dB, float* dC) {

    Copy_Atom<UniversalCopy<double>, double> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_32,_1>>{},  // 32x1 threads
                                      Layout<Shape< _1,_4>>{}); //  1x4 values

    auto gA = make_tensor(make_gmem_ptr(dA), make_shape(Int<32>{}, Int<4>{}), make_stride(Int<1>{}, Int<32>{}));      // (M,K)
    auto gB = make_tensor(make_gmem_ptr(dB), make_shape(Int<32>{}, Int<4>{}), make_stride(Int<1>{}, Int<32>{}));      // (N,K)
    auto gC = make_tensor(make_gmem_ptr(dC), make_shape(Int<32>{}, Int<4>{}), make_stride(Int<1>{}, Int<32>{}));      // (M,N)

    if (cute::thread0()) {
         print_tensor(gA);
    }


//    using Mma_atom = MMA_Atom<MMA_Traits<SM70_8x8x4_F16F16F16F16_NT>>;
//    using TiledMma = TiledMMA<
//        Mma_atom,
//        Layout<Shape<_1,_1,_1>>,  // 2x2x1 thread group
//        Layout<Shape<_1,_1,_1>>>; // 1x2x1 value group for 16x16x16 MMA and LDSM
//
//
//    TiledMma tiled_mma;
//
//    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
//
//    auto tAgA = thr_mma.partition_A(gA);
//    auto tBgB = thr_mma.partition_B(gB);
//    auto tCgC = thr_mma.partition_C(gC);
//
//    Tensor tArA  = thr_mma.partition_fragment_A(gA);
//    Tensor tBrB  = thr_mma.partition_fragment_B(gB);
//    Tensor tCrC  = thr_mma.partition_fragment_C(gC);
//
//
//    copy(tAgA, tArA);
//    copy(tBgB, tBrB);
//
//    gemm(tiled_mma, tArA, tBrB, tCrC);
//
//    if (threadIdx.x == 0) {
////        print_tensor(gA);
////        print_tensor(gB);
//        print_tensor(tAgA);
//        print_tensor(tBgB);
//        print_tensor(tCrC);
////        print_tensor(tArA);
////        print_tensor(tBrB);
////        print_tensor(tCrC);
//    }
//
//    copy(tCrC, tCgC);
////
////    printf("thread id = %d\n", threadIdx.x);
////    printf("tAgA = %f, tBgB = %f, tCgC = %f\n", tAgA[0], tBgB[0], tCgC[0]);
////    printf("tArA = %f, tBrB = %f, tCrC = %f\n", tArA[0], tBrB[0], tCrC[0]);


}


int main() {
    // initialise 2 32x32 matrices in float
    // convert them into 16b
    // define copy operation
    static int M = 32;
    static int N = 4;
    // Allocate memory on the host
    thrust::host_vector<half_t> hA(M*N);
    thrust::host_vector<half_t> hB(M*N);
    thrust::host_vector<float> hC(M*N);

    // Initialize matrices h_A and h_B with data
    for (int i=0; i<M*N; i++) {
        hA[i] = __float2half(i*1.0);
        hA[i] = __float2half(i*1.0);
        hC[i] = 0;
    }


    thrust::device_vector<half_t> dA = hA;
    thrust::device_vector<half_t> dB = hB;
    thrust::device_vector<float> dC = hC;

    //call mma
    mma_atom<<<1,32>>>(dA.data().get(), dB.data().get(), dC.data().get());

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }

//    hC = dC;
//    for (int i=0; i< N; i++){
//        for (int j=0; j< N; j++){
//            std::cout << hC[i*N+j] << " " ;
//        }
//        std::cout << std::endl;
//    }




    return 0;
}