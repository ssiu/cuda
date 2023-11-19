#include <cute/tensor.hpp>

using namespace cute;

int main() {
    // basic fma
    #if 0
    {
        auto tiled_mma = make_tiled_mma(UniversalFMA<float,float,float,float>{},
                                           Layout<Shape<_2,_2,_1>>{},
                                           Layout<Shape<_1, _1, _1>>{});

        print_latex(tiled_mma);
        //print_layout();
    }
    #endif

    #if 1
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