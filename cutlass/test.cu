#include <cute/layout.hpp>
#include <cute/tensor.hpp>
using namespace cute;


int main() {
    Layout layout = make_layout(make_shape (_2{}, _3{}),
                            make_stride(_3{}, _1{}));
    print_layout(layout);


//    Layout layout_2x4 = make_layout(make_shape (2, make_shape (2,2)),
//                              make_stride(4, make_stride(2,1)));
//
//    print_layout(layout_2x4)

//    Layout ThrID = Layout<Shape <_4, _2>,
//                       Stride<_1,_16>>;
//    print_layout(ThrID);
//
//    Layout CLayout = Layout<Shape <Shape <_2, _2,_2>, Shape <_2,_2, _2>>,
//                         Stride<Stride<_1,_16,_4>, Stride<_8,_2,_32>>>;
//    print(CLayout);
    return 0;
}

