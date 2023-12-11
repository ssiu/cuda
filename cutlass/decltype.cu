#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;


int main() {
    int kHeadDim = 64;
    int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    int kBlockM = 128;
    int kBlockN = 128;
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    return 0;
}

