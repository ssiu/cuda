
#include <cub/cub.cuh>


void sum_cub(float* d_in, float* d_out, int N) {
    // First call: Determine temporary device storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);

    // Allocate temporary storage
    void *d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Second call: Perform algorithm
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
}