
#include <cute/tensor.hpp>
using namespace cute;

int main()
{
    int A[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor a = make_tensor(&A[0], make_shape(Int<4>{}, Int<3>{}, Int<2>{}), make_stride(Int<6>{}, Int<2>{}, Int<1>{}));
    print_tensor(a); // (2, 2, 2)
    print_tensor(a(_,0,_)); // (2, 2)

    Tensor b = make_tensor(&A[0], make_shape(Int<4>{}, Int<3>{}, Int<2>{}), make_stride(Int<1>{}, Int<2>{}, Int<6>{}));
    print_tensor(b); // (2, 2, 2)
    print_tensor(b(_,0,_)); // (2, 2)
//     Tensor b = local_tile(a(_, 0, _), Shape<Int<1>, Int<2>>{},
//                           make_coord(_, 0));  // (1, 2, 2)
//     print_tensor(b);
//     printf("%d %d %d\n", b(0,0,1), b(0,1,0), b(0,0,1) + b(0,1,0));
//     b(0,0,1) = 10;
//     print_tensor(b);

}