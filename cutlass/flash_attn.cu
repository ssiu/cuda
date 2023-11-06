#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;


int main() {
    // flash_fwd_kernel.h
    // kBlockKSmem is either 64 or 32
    // if kBlockKSmem = 32 then kSwizzle = 2 else 3
    int kBlockKSmem = 64;
    int kSwizzle = 3;
    auto SmemLayoutAtomQ = composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{});

    print_layout(SmemLayoutAtomQ);

    int kBlockKSmem = 32;
    int kSwizzle = 2;
    auto SmemLayoutAtomQ = composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{});

    print_layout(SmemLayoutAtomQ);
    return 0;
}

