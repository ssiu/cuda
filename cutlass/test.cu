#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;


int main() {
    Layout layout = make_layout(make_shape (_2{}, _3{}),
                            make_stride(_3{}, _1{}));
    print_layout(layout);


    Layout layout_2x4 = make_layout(make_shape (2, make_shape (2,2)),
                              make_stride(4, make_stride(2,1)));

    print_layout(layout_2x4);

    Layout threadID = make_layout(make_shape(4,2),
                                make_stride(1,16));
    print_layout(threadID);

//    Layout Clayout = make_layout(make_shape(make_shape(2,2,2), make_shape(2,2,2)),
//                                make_stride(make_stride(1,16,4), make_stride(8,2,32)));
//    print_layout(CLayout);
    Layout layout1 = Layout<Shape <Shape <_4, _3>, _1>,
                         Stride<Stride<_3, _1>, _0>>{};
    print_layout(layout1);


    Layout CLayout = Layout<Shape <Shape <_2, _2,_2>, Shape <_2,_2, _2>>,
                         Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>>;
    print_layout(CLayout);
    //Layout flat_layout = flatten(layout);
    return 0;
}

