#include <cute/layout.hpp>
#include <cute/tensor.hpp>

using namespace cute;


int main() {
    // flash_fwd_kernel.h
    // kBlockKSmem is either 64 or 32
    // if kBlockKSmem = 32 then kSwizzle = 2 else 3

    // kBlockKSmem = 64
    // kSwizzle = 3
    auto SmemLayoutAtomQ = composition(Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{});

    print_layout(SmemLayoutAtomQ);

    // kBlockKSmem = 32
    // kSwizzle = 2
//    auto SmemLayoutAtomQ = composition(Swizzle<2, 3, 3>{},Layout<Shape<_8, _32>,Stride<_32, _1>>{});
//
//    //auto SmemLayoutQ = tile_to_shape(SmemLayoutAtomQ{},Shape<_4, _32>{});
//
//    print_layout(SmemLayoutAtomQ);
    //print_layout(SmemLayoutQ);
    return 0;
}

