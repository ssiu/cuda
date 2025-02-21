
#include <cute/tensor.hpp>
using namespace cute;

int main()
{
    // need to understand how to call the elements in a register tensor on partition_C when we tile the tensor core multiple times
    int C[16*16] = {0};
    for (int i = 0; i < 16*16; ++i) {
        C[i] = i;
    }


    auto layout_C = make_layout(make_shape (Int<16>{}, Int<16>{}),
                            make_stride(Int<8>{}, Int<1>{}));

    Tensor c = make_tensor(&C[0], layout_C);

    auto mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                Layout<Shape<_1, _1, _1>>{},
                                Tile<_16,_16,_8>{});
    ThrMMA thr_mma = mma.get_slice(0);

    Tensor tc = thr_mma.partition_C(c);


    print_tensor(c);


    print(tc);
    print_tensor(tc);
    printf("%d %d %d %d\n", int(size<0,0>(tc)), int(size<0,1>(tc)), int(size<1>(tc)), int(size<2>(tc)));

    printf("%d\n", tc(make_coord(0,0),0,0));

    print(tc(make_coord(_,0),_,_));
    print_tensor(tc(make_coord(_,0),_,_));

    for (int i=0; i< tc(make_coord(_,0),_,_).size(); i++) {
        printf("%d\n", tc(make_coord(_,0),_,_)[i]);
    }

}