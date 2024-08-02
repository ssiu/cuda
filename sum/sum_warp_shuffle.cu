// we launch N / 1024 blocks, with each block having 1024 threads
// each block computes a partial sum of a 1024 subarray
// each warp use warp shuffling to compute a single value
// we write the single value to shared memory
// the first warp will do reduction for the 32 values in shared memory, again using warp shuffling
// thread 0 then perform an atomic add to global memory

// naive matrix multiplication
#include <iostream>

__global__ void sum_warp_shuffle_kernel(float* d_in, float* d_out, int N){
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    float value;
    __shared__ float accum[32];

//    if (thread_id ==0 and blockIdx.x ==0) {
//        printf("%f\n", d_in[offset]);
//    }

    // 32 warps each do reduction of 32 values using warp shuffling
    value = d_in[offset];
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);

    // lane 0 of each warp writes result to shared memory
    if (lane_id == 0) {
        accum[warp_id] = value;
    }


    __syncthreads();

    // warp 0 perform reduction on the 32 values in shared memory, again using warp shuffling
    if (warp_id == 0) {
        value = accum[thread_id];

        value += __shfl_down_sync(0xffffffff, value, 16);
        value += __shfl_down_sync(0xffffffff, value, 8);
        value += __shfl_down_sync(0xffffffff, value, 4);
        value += __shfl_down_sync(0xffffffff, value, 2);
        value += __shfl_down_sync(0xffffffff, value, 1);


        if (thread_id == 0) {
            atomicAdd(&d_out[0], value);
        }
    }



}

void sum_warp_shuffle(float* d_in, float* d_out, int N) {
    dim3 dimGrid(N / 1024);
    dim3 dimBlock(1024);
    sum_warp_shuffle_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
}