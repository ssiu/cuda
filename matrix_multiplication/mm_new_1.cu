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


__global__ void mm_new_1(float* A, float* B, float* C, int N){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = by*TILE_WIDTH + ty;
    int col = bx*TILE_WIDTH + tx;
    A = &A[row*N];
    B = &B[col];
//    int row = by * TILE_WIDTH + ty;
//    int col = bx * TILE_WIDTH + tx;
    float sum = 0;
    for (int k=0; k<N; k++) {
        sum += A[k] * B[k*N];
    }

    C[row*N + col] = sum;

}