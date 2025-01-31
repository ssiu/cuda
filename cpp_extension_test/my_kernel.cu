#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void my_cuda_kernel(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = x[idx] * 2.0f;  // Example: Multiply each element by 2
    }
}

void launch_my_cuda_kernel(torch::Tensor x) {
    const int size = x.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    my_cuda_kernel<<<blocks, threads>>>(x.data_ptr<float>(), size);
}