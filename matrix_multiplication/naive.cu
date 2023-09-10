// naive kernel where each thread computes a single value


__global__ void matrix_multiplication(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i=0; i< N; i++){
            sum += A[row * N + i] * B[i*N + col];
        }
        C[row * N + col] = sum
    }
}

int main() {
    int N = 256; // Size of the square matrices
    int size = N * N * sizeof(float);

    // Allocate memory on the host
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices h_A and h_B with data
    for (int i=0; i< N*N; i++){
        A[i] = 1.0f;
        B[i] = 2.0f;
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
    dim3 dimGrid(N/16, N/16); // You can adjust this based on your GPU's capability
    dim3 dimBlock(16, 16);

    // Launch the matrix multiplication kernel
    matrixMultiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Copy the result matrix d_C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    for (int i=0; i< N; i++){
        for (int j=0; j< N; j++){
            std::cout << C[i*N+j] ;
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