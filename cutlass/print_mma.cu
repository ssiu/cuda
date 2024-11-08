#include <cute/tensor.hpp>

using namespace cute;

int main() {
    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{});
        print_latex(tiled_mma);
    }
    #endif

    #if 1
    {
        auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                            Layout<Shape<_1, _2, _1>>{}
                            );
        print_latex(tiled_mma);
    }
    #endif

    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                    Layout<Shape<_4, _1, _1>>{},  // 4x1x1 or 8x1x1 thread group
                    Layout<Shape<_1, _2, _1>>{}); // 1x2x1 or 1x2x2 value group for 16x16x16 MMA and LDSM
        print_latex(tiled_mma);
    }
    #endif
    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{});
        print_latex(tiled_mma);
    }
    #endif
    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                    composition(Swizzle<1,0,-1>{},
                                                Layout<Shape <Shape <_2,_2>,Shape <_2,_2>>,
                                                       Stride<Stride<_1,_4>,Stride<_2,_8>>>{}),
                                    Layout<Shape<_2,_2>>{},
                                    Tile<Layout<Shape<_4,_2,_2>,Stride<_1,_32,_8>>,
                                         Layout<Shape<_4,_2,_2>,Stride<_1,_32,_8>>>{});
         print_latex(tiled_mma);

    }
    #endif
    // basic fma
    #if 0
    {
        auto tiled_mma = make_tiled_mma(UniversalFMA<float,float,float,float>{},
                                           Layout<Shape<_2, _1, _1>>{},
                                           Layout<Shape<_1, _1, _2>>{});

        print_latex(tiled_mma);
        //print_layout();
    }
    #endif
    // custom fma
    #if 0
    {
//        template <class D, class A, class B, class C>
//        struct MMA_Traits<custom_32x32x1<D,A,B,C>>
//        {
//          using ElementDVal = D;
//          using ElementAVal = A;
//          using ElementBVal = B;
//          using ElementCVal = C;
//
//          // Logical shape of the MMA
//          using Shape_MNK = Shape<_32,_32,_1>;
//
//          // Logical thread id (tid) -> tidx
//          using ThrID   = Layout<_1>;
//
//          // (Logical thread id (tid), Logical value id (vid)) -> coord
//
//          // (tid,vid) -> (m,k)
//          using ALayout = Layout<Shape<_1,_32>>;
//          // (tid,vid) -> (n,k)
//          using BLayout = Layout<Shape<_1,_32>>;
//          // (tid,vid) -> (m,n)
//          using CLayout = Layout<Shape<_32,_32>, Stride<_1, _32>>;
//        };
//
//        auto tiled_mma = make_tiled_mma(custom_32x32x1<float,float,float,float>{},
//                                           Layout<Shape<_1,_1,_1>>{},
//                                           Layout<Shape<_1, _1, _1>>{});
//
//        print_latex(tiled_mma);
        //print_layout();
    }
    #endif

    #if 0
    {
        auto tiled_mma = make_tiled_mma(UniversalFMA<float,float,float,float>{},
                                           Layout<Shape<_2,_2,_1>>{},
                                           Layout<Shape<_1, _1, _2>>{});

        print_latex(tiled_mma);
        //print_layout();
    }
    #endif


    // flash attention
    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{});

        print_latex(tiled_mma);
    }
    #endif

    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_1,_1,_1>>{},
                                    Layout<Shape<_1, _2, _2>>{});

        print_latex(tiled_mma);
    }
    #endif

    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_4,_1,_1>>{},
                                    Layout<Shape<_1, _2, _2>>{});

        print_latex(tiled_mma);
    }
    #endif

    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                    Layout<Shape<_1, _1,_1>>{},      // 2x2 layout of atoms (threads)
                                    Layout<Shape<_2,_1,_1>>{});

        print_latex(tiled_mma);
    }
    #endif

    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                    Layout<Shape<_2,_2>>{},      // 2x2 layout of atoms (threads)
                                    Layout<Shape<_1,_1>>{},      // 1x1 layout of atoms (values)
                                    Tile<Layout<Shape <_4,_2>,   // Permutation in M
                                                Stride<_1,_8>>>{});

        print_latex(tiled_mma);
    }
    #endif

    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                    composition(Swizzle<1,0,-1>{},
                                                Layout<Shape <Shape <_2,_2>,Shape <_2,_2>>,
                                                       Stride<Stride<_1,_4>,Stride<_2,_8>>>{}),  // 4x4 layout of atoms (threads)
                                    Layout<Shape<_2,_2>>{},                                      // 2x2 layout of atoms (values)
                                    Tile<Layout<Shape<_4,_2,_2>,Stride<_1,_32,_8>>,              // Permutation in M
                                         Layout<Shape<_4,_2,_2>,Stride<_1,_32,_8>>>{});          // Permutation in N

        print_latex(tiled_mma);
    }
    #endif

    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                    Layout<Shape<_2,_2>>{},      // 2x2 tile of atoms (threads)
                                    Layout<Shape<_2,_2>>{},      // 2x2 tile of atoms (data)
                                    Tile<Layout<Shape <_4,_2>,
                                                Stride<_1,_16>>, // Permutation of M
                                         Layout<_1,_1>>{});         // Permutation of N

        print_latex(tiled_mma);
    }
    #endif

    #if 0
    {
        auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_1,_1,_1>>{},
                                    Layout<Shape<_1, _2, _2>>{});

        print_latex(tiled_mma);
    }
    #endif


    return 0;

}