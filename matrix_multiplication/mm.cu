#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

// 2048x2048x2048 gemm is 8.5 x 10^9 FLOPs
// shared memory per block is 48KB, which means 48*1024/4 = 12288 floats
// let each SM perform a 128x128 submatrix in the accumulator => 16x16 = 256 threadblocks in total
// at each SRAM load we do 128x128x32 so total SRAM need is 128x32x2x4 / 1024 = 32KB
// we are using 1024 threads per block so we are using 1024 threads to do 128x128x32 gemm
// so we are using 1024 threads to load 128x32x2 = 8192 numbers -> each threads load 8 numbers = 8*4 = 32
// using vectorized access each thread can load 4*4 = 16B numbers so need to do load twice
// at each load we need 1024*4*4 = 16KB registers

#define WIDTH 2048
#define OUTER_TILE_WIDTH 128
#define INNER_TILE_WIDTH 32
// This is needed for computing the addresses as float4 arrays
#define SHRINK_FACTOR = sizeof(float4) / sizeof(float)

__global__ void basic_mm(float* A, float* B, float* C, int N) {


    //global memory coalescing


    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    __shared__ float sA[OUTER_TILE_WIDTH*INNER_TILE_WIDTH];
    __shared__ float sB[OUTER_TILE_WIDTH*INNER_TILE_WIDTH];

    for (int TILE = 0; TILE < WIDTH/INNER_TILE_WIDTH; TILE++) {
        //global -> shared
        //each thread load 4 numbers for each A,B, so 8 numbers in total
        //After casting sA is now a float4 array of length 1024
        //global memory offset for A and B, we need to divide by 4 because we need the address wrt
        //to A,B as float4 arrays

        int offset_A = TILE * INNER_TILE_WIDTH / SHRINK_FACTOR + blockDim.y * blockIdx.y * OUTER_TILE_WIDTH * WIDTH / SHRINK_FACTOR;
        int offset_B = TILE * INNER_TILE_WIDTH / SHRINK_FACTOR + blockDim.x * blockIdx.x * OUTER_TILE_WIDTH * WIDTH / SHRINK_FACTOR;
        //as float4*, the matrix has size 2048x512 and the tile width is now 8
        reinterpret_cast<float4*>(sA)[tid] = reinterpret_cast<float4*>(A)[offset_A + tid * 512 / 8 + tid % 8 ];
        reinterpret_cast<float4*>(sB)[tid] = reinterpret_cast<float4*>(B)[offset_B + tid * 512 / 8 + tid % 8 ];
        __syncthreads();

        //shared -> register
         for (int i = 0; i < 32; i++) {
            float rA[4];
            float rB[4];
            //load A and B fragments
            for (int j = 0; j < 4; j++) {
                rA[j] = sA[(threadIdx.y * 4 + j) * INNER_TILE_WIDTH + i]
                rB[j] = sB[(threadIdx.x * 4 + j) * INNER_TILE_WIDTH + i]
            }

            int row = blockIdx.y * blockDim.y * OUTER_TILE_WIDTH + threadIdx.y;
            int col = blockIdx.x * blockDim.x * OUTER_TILE_WIDTH + threadIdx.x;

            float sum = 0.0f;

            for (int i = 0; i < 4; i++) {
                for (int j=0; j < 4; j++) {
                     sum += rA[i] * rB[j];
                }
            }

            C[(row + i) * OUTER_TILE_WIDTH + (col + j)] = sum;
            __syncthreads();
        }

    }

}



int main() {
    // Assume column major
    // Allocate memory on the host
    thrust::host_vector<float> hA(WIDTH * WIDTH);
    thrust::host_vector<float> hB(WIDTH * WIDTH);
    thrust::host_vector<float> hC(WIDTH * WIDTH);

    // Initialize matrices h_A and h_B with data
    for (int i=0; i< WIDTH * WIDTH; i++){
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