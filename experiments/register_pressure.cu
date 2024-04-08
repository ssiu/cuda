#include <iostream>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

__global__ void perform_float_operations(float* a, float* b, float* c) {
    c[0] = a[0] + b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] / b[0];
}

__global__ void perform_int_operations(int* a, int* b, int* c) {

    c[0] = a[0] + b[0];
    c[1] = a[0] * b[0];
    c[2] = a[0] / b[0];
    c[3] = a[0] % b[0];
    c[4] = a[0] >> 4;
    c[5] = a[0] & 3;

}



int main() {
    thrust::host_vector<float> a[1] = {5.0f};
    thrust::host_vector<float> b[1] = {3.0f};
    thrust::host_vector<float> c[6];
    thrust::host_vector<int> a_int[1] = {5};
    thrust::host_vector<int> b_int[1] = {3};
    thrust::host_vector<int> c_int[6];


    dim3 blockDim(1);
    dim3 gridDim(1);

    perform_float_operations<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(c.data()));
    perform_int_operations<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(c.data()));
    for (int i=0;i<6;i++){
        std::cout << c[i] << std::endl;
    }

    for (int i=0;i<6;i++){
        std::cout << c_int[i] << std::endl;
    }
    return 0;

}