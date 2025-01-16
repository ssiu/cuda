
#include <cute/tensor.hpp>
using namespace cute;

int main()
{
    int A[6] = {1, 2, 3, 4, 5, 6};
    Tensor a = make_tensor(A, 6);
    print_tensor(a);
}