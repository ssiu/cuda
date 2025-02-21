
#include <cute/tensor.hpp>
using namespace cute;

int main()
{
    int A[16*8] = {0};
    auto layout_A = make_layout(make_shape (Int<16>{}, Int<8>{}),
                            make_stride(Int<8>{}, Int<1>{}));

    Tensor b = make_tensor(&A[0], layout_A);

    auto mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{});
    ThrMMA thr_mma_O = mma.get_slice(0);



}