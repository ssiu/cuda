#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/mma.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;


// partition_fragment_A and partition_fragment_B often depend on the
//   layout of A and B and/or the thread_idx that is requesting the partition.
// For these reasons, they should not be used in a static context.
// See TiledMMA::get_slice(thr_idx).partition_fragment_A(tensorA) instead.
__global__ void mma_atom(half_t* dA, half_t* dB, float* dC) {

    auto gA = make_tensor(make_gmem_ptr(dA), make_shape(Int<8>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));      // (M,K)
    auto gB = make_tensor(make_gmem_ptr(dB), make_shape(Int<8>{}, Int<4>{}), make_stride(Int<1>{}, Int<8>{}));      // (N,K)
    auto gC = make_tensor(make_gmem_ptr(dC), make_shape(Int<8>{}, Int<8>{}), make_stride(Int<1>{}, Int<8>{}));      // (M,N)

    print(gA);
    print(gC);
    using Mma_atom = MMA_Atom<MMA_Traits<SM70_8x8x4_F16F16F16F16_NT>>;
    using TiledMma = TiledMMA<
        Mma_atom,
        Layout<Shape<_1,_1,_1>>,  // 2x2x1 thread group
        Layout<Shape<_1,_1,_1>>>; // 1x2x1 value group for 16x16x16 MMA and LDSM


    TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);

    auto tAgA = thr_mma.partition_A(gA);
    auto tBgB = thr_mma.partition_B(gB);
    auto tCgC = thr_mma.partition_C(gC);

    Tensor tArA  = thr_mma.partition_fragment_A(gA);
    Tensor tBrB  = thr_mma.partition_fragment_B(gB);
    Tensor tCrC  = thr_mma.partition_fragment_C(gC);


    copy(tAgA, tArA);
    copy(tBgB, tBrB);

    if (threadIdx.x == 0) {
        print_tensor(tAgA);
        print_tensor(tBgB);
        print_tensor(tCgC);
        print_tensor(tArA);
        print_tensor(tBrB);
        print_tensor(tCrC);
    }


    gemm(tiled_mma, tArA, tBrB, tCrC);

    copy(tCrC, tCgC);
//
//    printf("thread id = %d\n", threadIdx.x);
//    printf("tAgA = %f, tBgB = %f, tCgC = %f\n", tAgA[0], tBgB[0], tCgC[0]);
//    printf("tArA = %f, tBrB = %f, tCrC = %f\n", tArA[0], tBrB[0], tCrC[0]);


}


int main() {

    // Allocate memory on the host
    thrust::host_vector<half_t> hA(32);
    thrust::host_vector<half_t> hB(32);
    thrust::host_vector<float> hC(64);

    // Initialize matrices h_A and h_B with data
    for (int i=0; i<32; i++) {
        if (i % 8 == 0) {
            hA[i] = i;
        } else {
            hA[i] = 0;
        }
        if (i < 4) {
            hB[i] = 1;
        } else {
            hB[i] = 0;
        }
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

    hC = dC;
    for (int i=0; i< 8; i++){
        for (int j=0; j< 8; j++){
            std::cout << hC[i*8+j] << " " ;
        }
        std::cout << std::endl;
    }




    return 0;
}