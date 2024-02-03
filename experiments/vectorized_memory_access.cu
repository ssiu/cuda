#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


template <int N>
__global__ void device_copy_32_kernel(int* d_in, int* d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        d_out[i] = d_in[i];
    }
}


template <int N>
__global__ void device_copy_64_kernel(int* d_in, int* d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N/2; i += blockDim.x * gridDim.x) {
        reinterpret_cast<float2*>(d_out)[i] = reinterpret_cast<float2*>(d_in)[i]

    }
}

template <int N>
__global__ void device_copy_128_kernel(int* d_in, int* d_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
        reinterpret_cast<float4*>(d_out)[i] = reinterpret_cast<float4*>(d_in)[i]
    }
}


//128 threads loading an array of size N
int main() {
    int N = 1048576;
    thrust::host_vector<float> h_in(N);
    thrust::host_vector<float> h_out(N);

    for (int i=0; i< N; i++){
        h_in[i] = 1.0f;
    }

    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out = h_out;


    device_copy_32_kernel<1048576><<<128,8>>>(d_in.data().get(), d_out.data().get());
    device_copy_64_kernel<1048576><<<128,8>>>(d_in.data().get(), d_out.data().get());
    device_copy_128_kernel<1048576><<<128,8>>>(d_in.data().get(), d_out.data().get());


    return 0;
}