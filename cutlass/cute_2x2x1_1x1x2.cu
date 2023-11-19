#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/mma.hpp>

using namespace cute;


// partition_fragment_A and partition_fragment_B often depend on the
//   layout of A and B and/or the thread_idx that is requesting the partition.
// For these reasons, they should not be used in a static context.
// See TiledMMA::get_slice(thr_idx).partition_fragment_A(tensorA) instead.
__global__ void mma_atom(float* dA, float* dB, float* dC) {
    //printf("A = %f, B = %f\n", dA[0], dB[0]);
    //gemm(C[0], A[0], B[0], C[0]);

    auto gA = make_tensor(make_gmem_ptr(dA), make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}));      // (M,K)
    auto gB = make_tensor(make_gmem_ptr(dB), make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}));      // (N,K)
    auto gC = make_tensor(make_gmem_ptr(dC), make_shape(Int<2>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}));      // (M,N)

    using Mma_atom = MMA_Atom<MMA_Traits<UniversalFMA<float,float,float,float>>>;

    using TiledMma = TiledMMA<
      Mma_atom,
      Layout<Shape<_2,_2,_1>>,  // 2x2x1 thread group
      Layout<Shape<_1,_1,_1>>>; // 1x2x1 value group for 16x16x16 MMA and LDSM


    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);
    auto tCgC = thr_mma.partition_C(gC);

    Tensor tArA  = thr_mma.partition_fragment_A(gA);
    Tensor tBrB  = thr_mma.partition_fragment_B(gB);
    Tensor tCrC  = thr_mma.partition_fragment_C(gC);

    if (threadIdx.x == 0) {
        print_tensor(tAgA);
        print_tensor(tBgB);
        print_tensor(tCgC);
        print_tensor(tArA);
        print_tensor(tBrB);
        print_tensor(tCrC);
    }

    copy(tAgA, tArA);
    copy(tBgB, tBrB);

    gemm(tiled_mma, tArA, tBrB, tCrC);

    copy(tCrC, tCgC);

    printf("thread id = %d\n", threadIdx.x);
    printf("tAgA[0] = %f, tAgA[1] = %f, tBgB[0] = %f, tBgB[1] = %f, tCgC = %f\n", tAgA[0], tAgA[1], tBgB[0], tBgB[1], tCgC[0]);
    printf("tArA[0] = %f, tArA[1] = %f, tBrB[0] = %f, tBgB[1] = %f, tCrC = %f\n", tArA[0], tAgA[1], tBrB[0], tBgB[1], tCrC[0]);
//    print_tensor(gA);
//    auto rA = make_fragment_like(gA);
//    auto rB = make_fragment_like(gB);
//    auto rC = make_fragment_like(gC);
//
//
//    copy(gA, rA);
//    copy(gB, rB);
//
//    print_tensor(rA);
//
//    gemm(mma, rA, rB, rC);
//    copy(rC, gC);

//    printf("rA = %f, rB = %f, rC = %f\n", rA[0], rB[0], rC[0]);

}


int main() {

    // Allocate memory on the host
    thrust::host_vector<float> hA(4);
    thrust::host_vector<float> hB(4);
    thrust::host_vector<float> hC(4);

    // Initialize matrices h_A and h_B with data
    hA[0] = 1.0f;
    hA[1] = 2.0f;
    hA[2] = 3.0f;
    hA[3] = 5.0f;
    hB[0] = 7.0f;
    hB[1] = 11.0f;
    hB[2] = 13.0f;
    hB[3] = 17.0f;
    hC[0] = 0.0f;
    hC[1] = 0.0f;
    hC[2] = 0.0f;
    hC[3] = 0.0f;

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC = hC;

    //call mma
    mma_atom<<<1,4>>>(dA.data().get(), dB.data().get(), dC.data().get());

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }


    hC = dC;
    printf("C = %f %f\n %f %f \n", hC[0], hC[2], hC[1], hC[3]);



    return 0;
}