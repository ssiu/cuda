#include <iostream>
#include <typeinfo>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;


int main() {
    static constexpr int kHeadDim = 64;
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockM = 128;
    static constexpr int kBlockN = 128;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

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

    static constexpr int kSmemQAtomCount = size(SmemLayoutAtomQ{});
    static constexpr int kSmemQCount = size(SmemLayoutQ{});

    std::cout << "kSmemQAtomCount: " << kSmemQAtomCount << std::endl;
    std::cout << "kSmemQCount: " << kSmemQCount << std::endl;

    std::cout << "Type of SmemLayoutQ: " << typeid(SmemLayoutQ).name() << std::endl;


    return 0;
}

