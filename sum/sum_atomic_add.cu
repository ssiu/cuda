// we launch N / 1024 blocks, with each block having 1024 threads
// each block computes a partial sum of a 1024 subarray


// naive matrix multiplication
#include <iostream>

__global__ void sum_atomic_add_kernel(float* d_in, float* d_out, int N){
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_id = threadIdx.x;

    __shared__ float accum[1024];

//    if (thread_id ==0 and blockIdx.x ==0) {
//        printf("%f\n", d_in[offset]);
//    }
    accum[thread_id] = d_in[offset];

    __syncthreads();
    // now there are 1024 values that we need to add up
    // reduction tree
    // step  0: 1024 leaves, 512 threads, threads mod    2, stride   1
    // step  1:  512 leaves, 256 threads, threads mod    4, stride   2
    // step  2:  256 leaves, 128 threads, threads mod    8, stride   4
    // step  3:  128 leaves,  64 threads, threads mod   16, stride   8
    // step  4:   64 leaves,  32 threads, threads mod   32, stride  16
    // step  5:   32 leaves,  16 threads, threads mod   64, stride  32
    // step  6:   16 leaves,   8 threads, threads mod  128, stride  64
    // step  7:    8 leaves,   4 threads, threads mod  256, stride 128
    // step  8:    4 leaves,   2 threads, threads mod  512, stride 256
    // step  9:    2 leaves,   1 threads, threads mod 1024, stride 512
    // step 10:    1   leaf,       done!
    for (int stride = 1; stride < 1024; stride<<=1) {
        if (thread_id % (stride * 2) == 0) {
            accum[thread_id] += accum[thread_id + stride];
        }
        __syncthreads();
    }
//    if (thread_id ==0) {
//        printf("%f\n", accum[0]);
//    }
    if (thread_id == 0) {
        atomicAdd(&d_out[0], accum[0]);
    }
}

void sum_atomic_add(float* d_in, float* d_out, int N) {
    dim3 dimGrid(N / 1024);
    std::cout << N / 1024 << std::endl;
    dim3 dimBlock(1024);
    sum_atomic_add_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
}