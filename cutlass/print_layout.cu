#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;


int main() {


    #if 0
    {
        Layout layout = make_layout(make_shape (_8{}, _64{}),
                               make_stride(_64{}, _1{}));
        print_latex(layout);
    }
    #endif
    #if 1
    {
//        auto smem_atom = composition(Swizzle<2,0,3>{}, Layout<Shape<_4,_8>,Stride<_8,_1>>{});
//        print_layout(smem_atom);
        auto layout = composition(Swizzle<3, 3, 3>{},
                                    Layout<Shape<_8, _8>,
                                    Stride<_1, _8>>{});
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

