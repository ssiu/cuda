#include <torch/extension.h>

void launch_my_cuda_kernel(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_kernel", &launch_my_cuda_kernel, "My CUDA Kernel");
}