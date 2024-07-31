// one single threadblock in a grid
// we launch 1024 threads, each thread add a number with stride 1024


// naive matrix multiplication
#include <iostream>

#define FLOAT_4(pointer) reinterpret_cast<float4*>(&(pointer))[0]

__global__ void sum_vectorized_kernel(float* d_in, float* d_out, int N){
    //int threadId = threadIdx.x + blockDim.x * threadIdx.y
    int thread_id = threadIdx.x;

    float sum = 0;
    float reg[4];
    __shared__ float accum[1024];
    // loop through the array and add values with a stride of 4*1024
    for (int i = thread_id; i < N; i+=1024){
        FLOAT_4(reg[0]) = FLOAT_4(d_in[i*4]);

        #pragma unroll
        for (int j=0;j<4;j++){
            sum += reg[j];
        }
    }

//    if (thread_id ==0) {
//        printf("%f\n", sum);
//    }
    accum[thread_id] = sum;

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
    d_out[0] = accum[0];
}

void sum_vectorized(float* d_in, float* d_out, int N) {
    dim3 dimGrid(1);
    dim3 dimBlock(1024);
    sum_vectorized_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, N);
}