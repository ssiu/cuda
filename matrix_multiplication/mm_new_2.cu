#include <iostream>
// 32x32 threads computing a 32x32 block
#define TILE_WIDTH 32


__device__ void loadFromGmem(float* M, float* r, int addr){
    *r = M[addr];
}

__device__ void storeToSmem(float* r, float* sM, int addr){
    sM[addr] = *r;
}

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
    __shared__ float sA[TILE_WIDTH*TILE_WIDTH];
    __shared__ float sB[TILE_WIDTH*TILE_WIDTH];
    float rA;
    float rB;
    printf("hi");
    float sum = 0;
    for (int kBlock=0; kBlock<N/TILE_WIDTH; kBlock++){
//        sA[sPos] = A[gPos];
//        sB[sPos] = B[gPos];

        //load from gmem
        loadFromGmem(A, &rA, gPos);
        loadFromGmem(B, &rB, gPos);

        // store to sram
        storeToSmem(&rA, sA, sPos);
        storeToSmem(&rB, sB, sPos);

        //shift A,B pointers
        __syncthreads();
        A += TILE_WIDTH;
        B += TILE_WIDTH * N;
        // sync thread

        for (int k=0; k<TILE_WIDTH; k++) {
            sum += sA[ty*TILE_WIDTH + k] * sB[k*TILE_WIDTH + tx];
        }
        __syncthreads();

    }
    C[gPos] = sum;

}