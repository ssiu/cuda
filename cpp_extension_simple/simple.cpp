#include <torch/extension.h>
//using namespace std;

std::vector<torch::Tensor> Dummy(int const N) {
    std::vector<torch::Tensor> outputs;
    auto out = torch::zeros({1,2}, torch::dtype(torch::kFloat32));
    for (int n=0; n<N; n++)
        outputs.push_back(out.clone());
    return outputs;
}

std::vector<torch::Tensor> Dummy(int const N) {
    std::vector<torch::Tensor> outputs;
    auto out = torch::zeros({1,2}, torch::dtype(torch::kFloat32));
    for (int n=0; n<N; n++)
        outputs.push_back(out.clone());
    return outputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Dummy", &Dummy, "Dummy function.");
}