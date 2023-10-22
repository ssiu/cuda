#include <cute/layout.hpp>

using namespace cute;


int main() {
    Layout layout = make_layout(make_shape (_2{}, _3{}),
                            make_stride(_3{}, _1{}));
    print_layout(layout);
    return 0;
}

