// naive kernel where each thread computes a single value
#include <iostream>

__global__ void matrix_multiplication(float* A, float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i=0; i< N; i++){
            sum += A[row * N + i] * B[i*N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 1024; // Size of the square matrices
    int size = N * N * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices h_A and h_B with data
    for (int i=0; i< N*N; i++){
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }
    // Allocate memory on the device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices h_A and h_B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions for the kernel launch
    dim3 dimGrid(N/32, N/32); // You can adjust this based on your GPU's capability
    dim3 dimBlock(32, 32);

    // Launch the matrix multiplication kernel
    matrix_multiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "someKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        //goto Error; // Use appropriate error handling here
    }
    // Copy the result matrix d_C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i=0; i< N; i++){
        for (int j=0; j< N; j++){
            std::cout << h_C[i*N+j] << " " ;
        }
        std::cout << std::endl;
    }
    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}