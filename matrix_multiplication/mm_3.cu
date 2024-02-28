// matrix multiplication
// global memory coalescing
// shared memory blocking
// thread coarsening + vectorized memory access
// we will use 256 threads dimBlock(8, 32);

//    float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
//    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
//    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
//    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
//    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
#include <iostream>

#define TILE_WIDTH 32
__global__ void mm_3(float* A, float* B, float* C, int N){
    // offset in output C matrix
    int gRow_C = threadIdx.y + blockDim.y * blockIdx.y;
    int gCol_C = threadIdx.x + blockDim.x * blockIdx.x;

    // offset in shared memory;
    int sRow = threadIdx.y;
    int sCol = threadIdx.x;

    int gRow_A = gRow_C;
    int gCol_A;
    int gRow_B;
    int gCol_B = gCol_C;

    __shared__ float sA[TILE_WIDTH*TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH*TILE_WIDTH];

    // float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float sum[4] = {};
    for (int kTile=0; kTile < N/TILE_WIDTH; kTile++){
        //offset is row, kTile*TILE_WIDTH
        // reinterpret_cast<float2*>(d_out)[i]
        gCol_A = kTile*TILE_WIDTH / 4 + threadIdx.x;
        gRow_B = kTile*TILE_WIDTH / 4 + threadIdx.y;
        // bank conflict free G->S
//        reinterpret_cast<float4*>(sA)[sRow * TILE_WIDTH / 4 + sCol] = reinterpret_cast<float4*>(A)[gRow_A * N / 4 + gCol_A];
//        reinterpret_cast<float4*>(sB)[sRow * TILE_WIDTH / 4 + sCol] = reinterpret_cast<float4*>(B)[gRow_B * N / 4 + gCol_B];
//
//        __syncthreads();
//
//        for (int i=0; i<TILE_WIDTH; i++){
//            #pragma unroll
//            for (int j=0; j<TILE_WIDTH; j++) {
//                sum[j] += sA[sRow*TILE_WIDTH + (i + j)] * sB[(i+j) * TILE_WIDTH + sCol];
//            }
//        }
//        __syncthreads();
    }

//    reinterpret_cast<float4*>(C)[gRow_C * N / 4 + gCol_C] = reinterpret_cast<float4*>(sum)[0];
}