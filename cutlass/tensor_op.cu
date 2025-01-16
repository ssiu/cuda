
#include <cute/tensor.hpp>
using namespace cute;

int main()
{
    int A[6] = {1, 2, 3, 4, 5, 6};
    Tensor a = make_tensor(A, make_shape(Int<3>{},Int<2>{}), make_stride(Int<1>{},Int<3>{}));
    print_tensor(a);
}