// matrix multiplication
// global memory coalescing
// shared memory blocking
#include <iostream>

#define TILE_WIDTH 32
__global__ void mm_1(float* A, float* B, float* C, int N){
    //int threadId = threadIdx.x + blockDim.x * threadIdx.y
    // rows and columns that the thread compute the in output C matrix
    int cRow = threadIdx.y + blockDim.y * blockIdx.y;
    int cCol = threadIdx.x + blockDim.x * blockIdx.x;
    // rows and columns in shared memory;
    int sRow = threadIdx.y;
    int sCol = threadIdx.x;
    int gRow_A = cRow;
    int gCol_A;
    int gRow_B;
    int gCol_B = cCol;

    __shared__ float sA[TILE_WIDTH*TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH*TILE_WIDTH];

    // load into shared memory
    int sum = 0;
    for (int kTile=0; kTile < N/TILE_WIDTH; kTile++){
        //offset is row, kTile*TILE_WIDTH
        gCol_A = kTile*TILE_WIDTH + threadIdx.x;
        gRow_B = kTile*TILE_WIDTH + threadIdx.y;
        sA[sRow * TILE_WIDTH + sCol] = A[gRow_A * N + gCol_A];
        sB[sRow * TILE_WIDTH + sCol] = B[gRow_B * N + gCol_B];
        __syncthreads();

        for (int i=0; i<TILE_WIDTH; i++){
            sum += sA[sRow*TILE_WIDTH + i] * sB[i*TILE_WIDTH + sCol];
        }
        __syncthreads();
    }

    C[cRow*N + cCol] = sum;
}