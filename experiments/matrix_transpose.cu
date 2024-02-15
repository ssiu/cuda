#include <iostream>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>


// Naive
// Transpose a 1024x1024 row major matrix
// Each block has 1024 thread, handling a 32x32 matrix, with 32x32 thread blocks
// Each SM has 2048 thread, each thread loads 4B into shared memory, total shared memory = 2048*4B ~= 8KB
__global__ void naive_transpose(float* d_in, float* d_out, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    d_out[col * N + row] = d_in[row * N + col];

}

// Shared
// transpose a 1024x1024 row major matrix
// each block has 1024 thread, handling a 32x32 matrix, with 32x32 thread blocks
// each sm has 2048 thread, each thread loads 4B into shared memory, total shared memory = 2048*4B ~= 8KB
__global__ void shared_transpose(float* d_in, float* d_out, int N) {
    __shared__ float s[32*32];
    int col_s = threadIdx.x;
    int row_s = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    s[row_s * 32 + col_s] = d_in[row * N + col];
    __syncthreads();

    // transposed column and row
    col = blockIdx.y * blockDim.y + threadIdx.x;
    row = blockIdx.x * blockDim.x + threadIdx.y;

    d_out[row * N + col] = s[col_s * 32 + row_s];
}


// Bank conflict
__global__ void bank_conflict_transpose(float* d_in, float* d_out, int N) {

    //  G -> S
    //  a a a a
    //  b b b b
    //  c c c c
    //  d d d d

    //  S -> G (4-way bank conflict)
    //  a b c d
    //  a b c d
    //  a b c d
    //  a b c d

    //  padding
    //  G -> S
    //  a a a a
    //  o b b b
    //  b o c c
    //  c c o d
    //  d d d o

    //  permutation
    //  a b c d
    //  d a b c
    //  c d a b
    //  b c d a


    __shared__ float s[32*32];
    int col_s = threadIdx.x;
    int row_s = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    s[col_s * 32 + (row_s + col_s) % 32] = d_in[row * N + col];
    __syncthreads();

    col = blockIdx.y * blockDim.y + threadIdx.x;
    row = blockIdx.x * blockDim.x + threadIdx.y;

    d_out[row * N + col] = s[row_s * 32 + (row_s + col_s) % 32];

}
//
//// ILP
//// transpose a 1024x1024 row major matrix
//// each block has 1024 thread, handling a 64x64 matrix, with 16x16 thread blocks
//// each sm has 2048 thread, each thread loads 4*4B into shared memory, total shared memory = 2048*4*4B ~= 32KB
//__global__ void ILP_shared_transpose(float* d_in, float* d_out, int N) {
//    __shared__ float s[64*64];
//    int col_s = threadIdx.x;
//    int row_s = threadIdx.y;
//    int col = blockIdx.x * blockDim.x * 4 + threadIdx.x;
//    int row = blockIdx.y * blockDim.y * 4 + threadIdx.y;
//
//    s[row_s * 64 + col_s] = d_in[row * N + col];
//    s[row_s * 64 + col_s + 64] = d_in[row * N + col + 64];
//    s[(row_s + 64) * 64 + col_s] = d_in[(row + 64) * N + col];
//    s[(row_s + 64) * 64 + col_s + 64] = d_in[(row + 64) * N + col + 64];
//    __syncthreads();
//
//    d_out[row * N + col] = s[col_s * 64 + row_s];
//    d_out[row * N + col + 64] = s[col_s * 64 + row_s + 64];
//    d_out[(row + 64) * N + col] = s[(col_s + 64) * 64 + row_s]    ;
//    d_out[(row + 64) * N + col + 64] = s[(col_s + 64) * 64 + row_s + 64];
//}
//

//
////Bank conflict + ILP
//__global__ void ILP_shared_transpose(float* d_in, float* d_out, int N) {
//    __shared__ float s[64*64];
//    int col_s = threadIdx.x;
//    int row_s = threadIdx.y;
//    int col = blockIdx.x * blockDim.x * 4 + threadIdx.x;
//    int row = blockIdx.y * blockDim.y * 4 + threadIdx.y;
//
//    s[col_s * 64 + (row_s + col_s) % 64] = d_in[row * N + col];
//    s[col_s * 64 + (row_s + col_s) % 64 + 64] = d_in[row * N + col + 64];
//    s[(col_s + 64) * 64 + (row_s + col_s) % 64] = d_in[(row + 64) * N + col];
//    s[(col_s + 64) * 64 + (row_s + col_s) % 64 + 64] = d_in[(row + 64) * N + col + 64];
//    __syncthreads();
//
//    d_out[row * N + col] = s[col_s * 64 + row_s];
//    d_out[row * N + col + 64] = s[col_s * 64 + row_s + 64];
//    d_out[(row + 64) * N + col] = s[(col_s + 64) * 64 + row_s];
//    d_out[(row + 64) * N + col + 64] = s[(col_s + 64) * 64 + row_s + 64];
//}
//
//
//// vectorized
//// reinterpret_cast<float4*>(d_out)[i] = reinterpret_cast<float4*>(d_in)[i];
//// Shared
//// transpose a 1024x1024 row major matrix
//// each block has 1024 thread, handling a 64x64 matrix, with 16x16 thread blocks
//// each sm has 2048 thread, each thread loads 16B into shared memory, total shared memory = 2048*4B ~= 32KB
//__global__ void shared_transpose(float* d_in, float* d_out, int N) {
//    __shared__ float s[64*64/4];
//    int col_s = threadIdx.x;
//    int row_s = threadIdx.y;
//    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    int row = blockIdx.y * blockDim.y + threadIdx.y;
//    int tid = threadIdx.x + blockDim.x * threadIdx.y;
//    reinterpret_cast<float4*>(s)[(tid / 16) * 64 + tid % 16] = reinterpret_cast<float4*>(d_in)[(row + tid/16) * 256 + col/4 + tid % 16];
//    __syncthreads();
//
//    d_out[row * N + col] = s[col_s * 32 + row_s];
//}

int isSameMatrix(thrust::host_vector<float> A, thrust::host_vector<float> B, int N) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            if (A[i*N+j] != B[i*N+j]) {
                return 0;
            }
        }
    }
    return 1;
}

std::pair<thrust::host_vector<float>, thrust::host_vector<float>> generateMatrices(int N) {
    thrust::host_vector<float> A(N * N);
    thrust::host_vector<float> A_t(N * N);

    // Create random engine
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define distribution range
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    // Generate random matrix
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
//            float randomFloat = dis(gen);
//            A[i * N + j] = randomFloat;
//            A_t[j * N + i] = randomFloat;
            A[i * N + j] = static_cast<float>(i*N+j);
            A_t[j * N + i] = static_cast<float>(i*N+j);
        }
    }

    // Return both matrices
    return {A, A_t};
}

int main(int argc, char *argv[]) {
    // transpose an randon N x N matrix
    int N = std::stoi(argv[1]);

    auto [h_in, h_in_t] = generateMatrices(N);
    //thrust::host_vector<float> h_in(N * N);
    thrust::host_vector<float> h_out(N * N);
    //thrust::host_vector<float> h_tranpose(N * N);


    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out = h_out;

    //call mma
    //mma_atom<<<1,1>>>(dA.data().get(), dB.data().get(), dC.data().get());

    dim3 dimGrid(32, 32); // You can adjust this based on your GPU's capability
    dim3 dimBlock(32, 32);


    // Launch the matrix multiplication kernel
    naive_transpose<<<dimGrid, dimBlock>>>(d_in.data().get(), d_out.data().get(), N);
    shared_transpose<<<dimGrid, dimBlock>>>(d_in.data().get(), d_out.data().get(), N);
    bank_conflict_transpose<<<dimGrid, dimBlock>>>(d_in.data().get(), d_out.data().get(), N);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }

    h_out = d_out;

    if (isSameMatrix(h_out, h_in_t, N) == 0) {
        printf("Wrong answer\n");
    }

//    for (int i=0; i < 100; i++){
//        printf("%f %f %f\n", h_in[i], h_out[i], h_in_t[i]);
//    }


    return 0;
}