#include <torch/extension.h>

//torch::Tensor flash_fwd_v0(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

//torch::Tensor flash_fwd_v1(torch::Tensor q, torch::Tensor k, torch::Tensor v,
//                            int batch_size, int seq_len, int num_heads, int head_dim);

torch::Tensor flash_fwd_v2(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                            int batch_size, int seq_len, int num_heads, int head_dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //m.def("flash_fwd_v0", &flash_fwd_v0, "flash fwd v0");
    //m.def("flash_fwd_v1", &flash_fwd_v1, "flash fwd v1");
    m.def("flash_fwd_v2", &flash_fwd_v2, "flash fwd v2");
}