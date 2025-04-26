#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;


int main() {
    #if 1
    {
        using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;

        using TiledMma_S = TiledMMA<
            MMA_Atom_Arch,
            Layout<Shape<_2, _4, _1>>,
            Tile<_32, _32, _8>>;

        int q_32_ptr[32*128];
        int q_64_ptr[64*128];
        Tensor q_32 = make_tensor(&q_32_ptr[0], make_shape(Int<32>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));
        Tensor q_64 = make_tensor(&q_64_ptr[0], make_shape(Int<64>{}, Int<128>{}), make_stride(Int<128>{}, Int<1>{}));

        TiledMma_S tiled_mma_S;
        ThrMMA thr_mma_S = tiled_mma_S.get_slice(0);
        Tensor tSgQ_32 = thr_mma_S.partition_A(q_32);
        Tensor tSgQ_64 = thr_mma_S.partition_A(q_64);
        print(tSgQ_32);
        print("\n=========================\n");
        print(tSgQ_64);

    }
    #endif

    #if 0
    {
//         using SmemLayoutAtomA = decltype(composition(
//             Swizzle<3, 3, 3>{},
//             make_layout(make_shape(Int<32>{}, Int<8>{}),
//                         make_stride(Int<1>{}, Int<32>{}))));
//         using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
//                                                    make_shape(Int<128>{}, Int<128>{})));
//         SmemLayoutA sA_layout;


        // column major
//         auto layout_atom = composition(Swizzle<3, 3, 3>{},
//                                     Layout<Shape<_64,_16>,
//                                     Stride<_1, _64>>{});



        // row major
        // fp 32
        auto layout_atom = composition(Swizzle<3, 3, 3>{},
                                    Layout<Shape<_16,_32>,
                                    Stride<_32, _1>>{});
// // fp16
//         auto layout_atom = composition(Swizzle<3, 3, 3>{},
//                                     Layout<Shape<_16,_64>,
//                                     Stride<_64, _1>>{});





        print_layout(layout_atom);


        //         auto layout = tile_to_shape(layout_atom,
        //                                        make_shape(Int<64>{}, Int<16>{}));
        //print_layout(layout);

    }
    #endif

    #if 0
    {

        using SmemLayoutAtomA = decltype(composition(
            Swizzle<3, 3, 3>{},
            make_layout(make_shape(Int<64>{}, Int<16>{}),
                        make_stride(Int<1>{}, Int<64>{}))));
        using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtomA{},
                                                   make_shape(Int<128>{}, Int<32>{})));
        SmemLayoutA sA_layout;

        using SmemLayoutAtomB = decltype(composition(
            Swizzle<3, 3, 3>{},
            make_layout(make_shape(Int<32>{}, Int<32>{}),
                        make_stride(Int<32>{}, Int<1>{}))));
        using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtomB{},
                                                   make_shape(Int<128>{}, Int<32>{})));
        SmemLayoutB sB_layout;

        print_layout(sA_layout);
        print_layout(sB_layout);


//         auto SmemLayoutAtom = decltype(composition(
//                                 Swizzle<3, 3, 3>{},
//                                 make_layout(make_shape(Int<32>{}, Int<8>{}),
//                                             make_stride(Int<1>{}, Int<128>{}))));
//
//         auto SmemLayout = tile_to_shape(SmemLayoutAtom{}, make_shape(Int<128>{}, Int<32>{}));




//         auto layout1 = composition(Swizzle<3, 3, 2>{},
//                                     Layout<Shape<_128, _32>,
//                                     Stride<_1, _128>>{});
//
//         print_layout(layout1);
//         auto layout2 = composition(Swizzle<3, 3, 3>{},
//                                     Layout<Shape<_128, _32>,
//                                     Stride<_32, _1>>{});
//
//
//         print_layout(layout2);
    }
    #endif

    #if 0
    {
//        auto smem_atom = composition(Swizzle<2,0,3>{}, Layout<Shape<_4,_8>,Stride<_8,_1>>{});
//        print_layout(smem_atom);
        auto layout = composition(Swizzle<2, 3, 3>{},
                                    Layout<Shape<_128, _32>,
                                    Stride<_32, _1>>{});
        print_latex(layout);
    }
    #endif


    #if 0
        auto copy_atom = Layout<Shape<_8, _4>,
                                Stride<_1,_8>>{};

        auto tile_shape = Shape<_32, _32>{};

        auto copy_layout =  tile_to_shape(copy_atom, tile_shape);

        print_latex(copy_layout);
    #endif


    #if 0
    {
        Layout layout = make_layout(make_shape (_8{}, _64{}),
                               make_stride(_64{}, _1{}));
        print_latex(layout);
    }
    #endif
    #if 0
    {
//        auto smem_atom = composition(Swizzle<2,0,3>{}, Layout<Shape<_4,_8>,Stride<_8,_1>>{});
//        print_layout(smem_atom);
        auto layout = composition(Swizzle<2, 3, 3>{},
                                    Layout<Shape<_128, _8>,
                                    Stride<_8, _1>>{});
        print_layout(layout);
    }
    #endif


//    print_layout(layout);
//    Layout layout = make_layout(make_shape (_2{}, _3{}),
//                            make_stride(_3{}, _1{}));
//    print_layout(layout);
//
//
//    Layout layout_2x4 = make_layout(make_shape (2, make_shape (2,2)),
//                              make_stride(4, make_stride(2,1)));
//
//    print_layout(layout_2x4);
//
//    Layout threadID = make_layout(make_shape(4,2),
//                                make_stride(1,16));
//    print_layout(threadID);
//
////    Layout Clayout = make_layout(make_shape(make_shape(2,2,2), make_shape(2,2,2)),
////                                make_stride(make_stride(1,16,4), make_stride(8,2,32)));
////    print_layout(CLayout);
//    Layout layout1 = Layout<Shape <Shape <_4, _3>, _1>,
//                         Stride<Stride<_3, _1>, _0>>{};
//    print_layout(layout1);
//
//
//    Layout CLayout = Layout<Shape <Shape <_2, _2,_2>, Shape <_2,_2, _2>>,
//                         Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>>{};
//    print_layout(CLayout);
//    //Layout flat_layout = flatten(layout);
//




//    auto smem_atom = composition(Swizzle<2,0,3>{}, Layout<Shape<_4,_8>,Stride<_8,_1>>{});
//    print_layout(smem_atom);
//    auto atom = composition(Swizzle<1,0,-1>{}, Layout<Shape <Shape <_2,_2>,Shape <_2,_2>>, Stride<Stride<_1,_4>,Stride<_2,_8>>>{});
//    print_layout(atom);


// from cutlass sm70
//    Layout SM70_QuadPair = Layout<Shape <_4, _2>,
//                             Stride<_1,_16>>{};
//    // (T8,V4) -> (M8,K4)
//    Layout SM70_8x4_Row  = Layout<Shape <_8,_4>,
//                                 Stride<_1,_8>>{};
//    // (T8,V4) -> (M8,K4)
//    Layout SM70_8x4_Col  = Layout<Shape <Shape <_4,_2>,_4>,
//                                 Stride<Stride<_8,_4>,_1>>{};
//    // (T8,V8) -> (M8,N8)
//    Layout SM70_8x8_16b  = Layout<Shape <_8,_8>,
//                                 Stride<_1,_8>>{};
//    // (T8,V8) -> (M8,N8)
//    Layout SM70_8x8_32b  = Layout<Shape <Shape <_2, _2,_2>,Shape <_2,_2, _2>>,
//                                 Stride<Stride<_1,_16,_4>,Stride<_8,_2,_32>>>{};
//
//    print_layout(SM70_QuadPair);
//    print_layout(SM70_8x4_Row);
//    print_layout(SM70_8x4_Col);
//    print_layout(SM70_8x8_16b);
//    print_layout(SM70_8x8_32b);


    return 0;
}

