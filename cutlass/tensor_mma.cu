
#include <cute/tensor.hpp>
using namespace cute;

int main()
{

    // need to understand how to call the elements in a register tensor on partition_C when we tile the tensor core multiple times
    int C[64*64] = {0};
    for (int i = 0; i < 64*64; ++i) {
        C[i] = i;
    }


    auto layout_C = make_layout(make_shape (Int<64>{}, Int<64>{}),
                            make_stride(Int<64>{}, Int<1>{}));

    Tensor c = make_tensor(&C[0], layout_C);

    auto mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                Layout<Shape<_2, _4, _1>>{},
                                Tile<_64,_16,_8>{});
    ThrMMA thr_mma = mma.get_slice(0);

    Tensor tc = thr_mma.partition_C(c);


    //print_tensor(c);

    // ptr[32b](0x7ea2408d8910) o ((_2,_2),_2,_2):((_1,_512),_2048,_32)
    print(tc);
    print_tensor(tc);
    for (int i=0;i<2;i++) {
        for (int j=0;j<2;j++) {
            print_tensor(tc(make_coord(_,j),i,_));
        }
    }

//     printf("%d\n", tc(make_coord(0,0),0,0));
//     printf("%d\n", tc(make_coord(0,0),0,1));
//     printf("%d\n", tc(make_coord(1,0),0,0));
//     printf("%d\n", tc(make_coord(1,0),0,1));
//     printf("%d %d %d %d\n", int(size<0,0>(tc)), int(size<0,1>(tc)), int(size<1>(tc)), int(size<2>(tc)));
//
//     printf("%d\n", tc(make_coord(0,0),0,0));
//
//     print(tc(make_coord(_,0),_,_));
//     print_tensor(tc(make_coord(_,0),_,_));
//
//     printf("printing the elements\n");
//     for (int i=0; i< tc(make_coord(_,0),_,_).size(); i++) {
//         printf("%d\n", tc(make_coord(_,0),_,_)[i]);
//     }
//     print("=====\n");
//     for (int i=0; i< tc(make_coord(_,1),_,_).size(); i++) {
//         printf("%d\n", tc(make_coord(_,1),_,_)[i]);
//     }
//     print("=====\n");
//     for (int i=0; i< tc(make_coord(0,_),_,_).size(); i++) {
//         printf("%d\n", tc(make_coord(0,_),_,_)[i]);
//     }
//     print("=====\n");
//     for (int i=0; i< tc(make_coord(1,_),_,_).size(); i++) {
//         printf("%d\n", tc(make_coord(1,_),_,_)[i]);
//     }
//     print("=====\n");
//
//     Tensor td = tc;
//     print(td);

}