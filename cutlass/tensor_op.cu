
#include <cute/tensor.hpp>
using namespace cute;

int main()
{
    int A[6] = {1, 2, 3, 4, 5, 6, 7, 8};
    Tensor a = make_tensor(&A[0], make_shape(Int<2>{}, Int<2>{}, Int<2>{}));
    print_tensor(a);
}