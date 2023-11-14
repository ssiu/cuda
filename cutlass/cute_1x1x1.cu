#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma.hpp>
// using cute machinery to for 1x1x1


// initial a 1x1x1 matrices


// copy 1x1x1 matrix
// copy traits
using namespace cute;

__global__ void mma(float* dA, float* dB, float* dC) {
    printf("A = %f, B = %f\n", dA[0], dB[0]);
    //gemm(C[0], A[0], B[0], C[0]);

    auto mA = make_tensor(make_gmem_ptr(dA), make_shape(1,1), make_stride(1, 1));      // (M,K)
    auto mB = make_tensor(make_gmem_ptr(dB), make_shape(1,1), make_stride(1, 1));      // (N,K)
    auto mC = make_tensor(make_gmem_ptr(dC), make_shape(1,1), make_stride(1, 1);      // (M,N)
//
//    rA = make_fragment_like(mA);
//    rB = make_fragment_like(mB);
//    rC = make_fragment_like(mC);
//
//
//    algorithm::gemm(rA, rB, rC);

}
// do mma
// mma traits


int main() {

    // Allocate memory on the host
    thrust::host_vector<float> hA(1);
    thrust::host_vector<float> hB(1);
    thrust::host_vector<float> hC(1);

    // Initialize matrices h_A and h_B with data
    hA[0] = 2.0f;
    hB[0] = 3.0f;
    hC[0] = 0.0f;

    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC = hC;

    //call mma
    mma<<<1,1>>>(dA.data().get(), dB.data().get(), dC.data().get());

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }


    hC = dC;
    printf("C = %f \n", hC[0]);



    return 0;
}