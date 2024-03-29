#include <iostream>

#define TILE_WIDTH 32
#define COARSE_FACTOR 2
__global__ void matrix_multiplication(float* A, float* B, float* C, int WIDTH) {

    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;


    int row = by * TILE_WIDTH + ty;
    // the starting column that this thread computes
    int col_start = bx * TILE_WIDTH * COARSE_FACTOR + tx;


    float s[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; c++) {
        s[c] = 0.0f;
    }

    for ( int tile = 0; tile < WIDTH / TILE_WIDTH; tile++){

        Ads[ty][tx] = A[row * WIDTH + tile * TILE_WIDTH + tx];

        for (int c = 0; c < COARSE_FACTOR; c++) {
            int col = col_start + TILE_WIDTH * c;
            Bds[ty][tx] = B[WIDTH * (tile * TILE_WIDTH + ty) + col];
            __syncthreads();

            for ( int k = 0; k < TILE_WIDTH; k++){
                s[c] += Ads[ty][k] * Bds[k][ty];
            }
            __syncthreads();

        }

    }
    for (int c = 0; c < COARSE_FACTOR; c++) {

        int col = col_start + TILE_WIDTH * c;

        if (row < WIDTH && col < WIDTH) {
            C[row * WIDTH + col] = s[c];
        }

    }

}


int main() {
    int N = 2048; // Size of the square matrices
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
    //
    dim3 dimGrid(N/(32 * COARSE_FACTOR), N/32); // You can adjust this based on your GPU's capability
    dim3 dimBlock(32, 32);

    // Launch the matrix multiplication kernel
    matrix_multiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Copy the result matrix d_C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
//    for (int i=0; i< N; i++){
//        for (int j=0; j< N; j++){
//            std::cout << h_C[i*N+j] << " " ;
//        }
//        std::cout << std::endl;
//    }
    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}