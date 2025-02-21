
#include <cute/tensor.hpp>
using namespace cute;

int main()
{
    // need to understand how to call the elements in a register tensor on partition_C when we tile the tensor core multiple times
    int A[16*8] = {0};
    auto layout_A = make_layout(make_shape (Int<16>{}, Int<8>{}),
                            make_stride(Int<8>{}, Int<1>{}));

    Tensor a = make_tensor(&A[0], layout_A);

    auto mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                Layout<Shape<_1, _1, _1>>{},
                                Tile<_16,_16,_8>{});
    ThrMMA thr_mma = mma.get_slice(0);

    Tensor ta = thr_mma.partition_A(a);

    print(ta);
    print_tensor(ta);
    printf("%d %d %d %d\n", int(size<0,0>(ta)), int(size<0,1>(ta)), int(size<1>(ta)), int(size<2>(ta)));
}