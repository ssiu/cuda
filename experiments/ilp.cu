
// This kernel
// variables: number of threads
//            number of ILP

#define N 1024

__global__ void arithmetic_kernel(int num_instructions) {
    int a = 1;

    #pragma unroll 1
    for (int i = 0; i < N; i ++) {
        a = a * 1 + 1;
    }
}


int main(){

    arithmetic_kernel<<<1024, 128>>>(1);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }
    return 0;

}