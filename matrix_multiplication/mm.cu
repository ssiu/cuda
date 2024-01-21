#include <thrust/host_vector.h>
#include <thrust/device_vector.h>



int main() {
    const int N = 2048:
    // Allocate memory on the host
    thrust::host_vector<float> hA(N*N);
    thrust::host_vector<float> hB(N*N);
    thrust::host_vector<float> hC(N*N);

    // Initialize matrices h_A and h_B with data
    for (int i=0; i< N*N; i++){
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }


    thrust::device_vector<float> dA = hA;
    thrust::device_vector<float> dB = hB;
    thrust::device_vector<float> dC = hC;

    //call mma
    //mma_atom<<<1,1>>>(dA.data().get(), dB.data().get(), dC.data().get());

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }


    hC = dC;
    printf("C = %f \n", hC[0]);

    return 0;
}