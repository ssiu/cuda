#include <torch/extension.h>

void launch_my_cuda_kernel(torch::Tensor x);
torch::Tensor mm_new_8(torch::Tensor a, torch::Tensor b);
//torch::Tensor gemm_register_pipelining_256(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_kernel", &launch_my_cuda_kernel, "My CUDA Kernel");
    m.def("mm_new_8", &mm_new_8, "gemm kernel");
    //m.def("gemm_register_pipelining_256", &gemm_register_pipelining_256, "cutlass gemm kernel");
}