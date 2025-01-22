
#include <cute/tensor.hpp>
using namespace cute;

int main()
{
    int A[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor a = make_tensor(&A[0], make_shape(Int<2>{}, Int<2>{}, Int<2>{}), make_stride(Int<4>{}, Int<2>{}, Int<1>{}));
    print_tensor(a); // (2, 2, 2)
    print_tensor(a(_,0,_)); // (2, 2)

    Tensor b = local_tile(a(_, 0, _), Shape<Int<1>, Int<2>>{},
                          make_coord(_, 0));  // (1, 2, 2)
    print_tensor(b);
    printf("%d", (b(0,0,1));


}