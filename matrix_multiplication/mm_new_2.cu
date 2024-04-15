#include <iostream>
// 32x32 threads computing a 32x32 block
#define TILE_WIDTH 32


//__device__ void loadFromGmem(){
//
//}
//
//__device__ void storeToSmem(){
//
//}
//
//__device__ void loadFromSmem(){
//
//
//}
//
//__device__ void computeOuterProduct(){
//
//}


__global__ void mm_new_2(float* A, float* B, float* C, int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gRow = by*TILE_WIDTH;
    int gCol = bx*TILE_WIDTH;
    int gPos = ty*N+tx;
    int sPos = ty*TILE_WIDTH + tx;
    A = &A[gRow*N];
    B = &B[gCol];
    C = &C[gRow*N + gCol];
    __shared__ float sA[1024];
    __shared__ float sB[1024];

    float sum = 0;
    for (int kBlock=0; kBlock<N/TILE_WIDTH; kBlock++){
        //load from gmem
        sA[sPos] = A[gPos];
        sB[sPos] = B[gPos];

        // store to sram

        //shift A,B pointers
        __syncthreads();
        A += TILE_WIDTH;
        B += TILE_WIDTH * N;
        // sync thread


        for (int k=0; k<N; k++) {
            sum += sA[ty*TILE_WIDTH + k] * sB[k*TILE_WIDTH + tx];
        }
    }
    C[gPos] = sum;

}