
#include <cub/cub.cuh>


void sum_cub(float* d_in, float* d_out, int N) {

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
//    cudaMalloc(&d_temp_storage, temp_storage_bytes);
//    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
//    cudaFree(d_temp_storage);

}