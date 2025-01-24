#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "flash_fwd_v0.cu"
#include "utils.cuh"

int main(int argc, char** argv)
{
    int batch_size = 1;
    if (argc >= 2)
    sscanf(argv[1], "%d", &batch_size);

    int seq_len = 128;
    if (argc >= 3)
    sscanf(argv[2], "%d", &seq_len);

    int num_heads = 1;
    if (argc >= 4)
    sscanf(argv[3], "%d", &num_heads);

    int head_dim = 128;
    if (argc >= 5)
    sscanf(argv[3], "%d", &head_dim);


    cute::device_init(0);

    thrust::host_vector<half_t> h_Q = generateRandomMatrix<half_t> (batch_size * seq_len * num_heads * head_dim);
    thrust::host_vector<half_t> h_K = generateRandomMatrix<half_t> (batch_size * seq_len * num_heads * head_dim);
    thrust::host_vector<half_t> h_V = generateRandomMatrix<half_t> (batch_size * seq_len * num_heads * head_dim);

    thrust::host_vector<float> h_O(batch_size * seq_len * num_heads * head_dim, 0.0f);



    thrust::device_vector<half_t> d_Q = h_Q;
    thrust::device_vector<half_t> d_K = h_K;
    thrust::device_vector<half_t> d_V = h_V;
    thrust::device_vector<float> d_O = h_O;

    flash_fwd_v0(d_Q, d_K, d_V, d_O, batch_size, seq_len, num_heads, head_dim);

    h_O = d_O;

    return 0;
}



