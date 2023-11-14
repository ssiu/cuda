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

__global__ void mma(float* A, float* B, float* C) {
    printf("A = %f, B = %f\n", A[0], B[0]);
    //gemm(C[0], A[0], B[0], C[0]);

}
// do mma
// mma traits


int main() {

    // Allocate memory on the host
    thrust::host_vector<float> h_A(1);
    thrust::host_vector<float> h_B(1);
    thrust::host_vector<float> h_C(1);

    // Initialize matrices h_A and h_B with data
    h_A[0] = 2.0f;
    h_B[0] = 3.0f;
    h_C[0] = 0.0f;

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C = h_C;

    //call mma
    mma<<<1,1>>>(d_A.data().get(), d_B.data().get(), d_C.data().get());

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }


    h_C = d_C;
    printf("C = %f \n", h_C[0]);



    return 0;
}