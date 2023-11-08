#include <cute/tensor.hpp>

using namespace cute;

int main() {

    auto tiled_mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                                    Layout<Shape<_2,_2>>{},      // 2x2 layout of atoms (threads)
                                    Layout<Shape<_1,_1>>{},      // 1x1 layout of atoms (values)
                                    Tile<Layout<Shape <_4,_2>,   // Permutation in M
                                                Stride<_1,_8>>>{});

    print_latex(tiled_mma);
    return 0;

}