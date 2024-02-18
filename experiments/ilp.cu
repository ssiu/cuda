#include <iostream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

// This kernel
// variables: number of threads
//            number of ILP

// #define N 1000000000

__global__ void naive_transpose(float* d_in, float* d_out, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    d_out[col * N + row] = d_in[row * N + col];

}

//__global__ void arithmetic_kernel_1(float x) {
//    int idx = blockDim.x * blockIdx.x + threadIdx.x;
//
//    int a = idx + x;
//
//    //#pragma unroll 1
////    for (int i = 0; i < N; i ++) {
////        a = a * 1 + 1;
////    }
//}


//__global__ void arithmetic_kernel_2() {
//    int a = 1;
//    int b = 1;
//
//    //#pragma unroll 1
//    for (int i = 0; i < N; i ++) {
//        a = a * 1 + 1;
//        b = b * 1 + 1;
//    }
//}
//
//__global__ void arithmetic_kernel_3() {
//    int a = 1;
//    int b = 1;
//    int c = 1;
//
//    //#pragma unroll 1
//    for (int i = 0; i < N; i ++) {
//        a = a * 1 + 1;
//        b = b * 1 + 1;
//        c = c * 1 + 1;
//    }
//}
//
//__global__ void arithmetic_kernel_4() {
//    int a = 1;
//    int b = 1;
//    int c = 1;
//    int d = 1;
//
//    //#pragma unroll 1
//    for (int i = 0; i < N; i ++) {
//        a = a * 1 + 1;
//        b = b * 1 + 1;
//        c = c * 1 + 1;
//        d = d * 1 + 1;
//    }
//}
//
//__global__ void arithmetic_kernel_5() {
//    int a = 1;
//    int b = 1;
//    int c = 1;
//    int d = 1;
//    int e = 1;
//
//    #pragma unroll
//    for (int i = 0; i < N; i ++) {
//        a = a * 1 + 1;
//        b = b * 2 + 1;
//        c = c * 3 + 1;
//        d = d * 4 + 1;
//        e = e * 5 + 1;
//    }
//}

thrust::host_vector<float> generateMatrix(int N) {
    thrust::host_vector<float> A(N * N);

    // Create random engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define distribution range
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // Generate random matrix
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {//
            A[i * N + j] = static_cast<float>(i*N+j);
        }
    }

    // Return both matrices
    return A;
}



int main(){
    int N = 1024;

//    arithmetic_kernel_1<<<numBlocks, numThreads>>>();
//    arithmetic_kernel_2<<<numBlocks, numThreads>>>();
//    arithmetic_kernel_3<<<numBlocks, numThreads>>>();
//    arithmetic_kernel_4<<<numBlocks, numThreads>>>();
//    arithmetic_kernel_5<<<numBlocks, numThreads>>>();


    auto h_in = generateMatrix(N);
    //thrust::host_vector<float> h_in(N * N);
    thrust::host_vector<float> h_out(N * N);
    //thrust::host_vector<float> h_tranpose(N * N);


    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out = h_out;

    //call mma
    //mma_atom<<<1,1>>>(dA.data().get(), dB.data().get(), dC.data().get());

    dim3 dimGrid(32, 32); // You can adjust this based on your GPU's capability
    dim3 dimBlock(32, 32);

    naive_transpose<<<dimGrid, dimBlock>>>(d_in.data().get(), d_out.data().get(), N);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }

    h_out = d_out;

    return 0;

}