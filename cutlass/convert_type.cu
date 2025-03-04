
#include <cute/tensor.hpp>


#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>


using namespace cute;

int main()
{
// template <typename To_type, typename Engine, typename Layout>
// __forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
//     using From_type = typename Engine::value_type;
//     constexpr int num_element = decltype(size(tensor))::value;
//     cutlass::NumericArrayConverter<To_type, From_type, num_element> convert_op;
//     // HACK: this requires tensor to be "contiguous"
//     auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, num_element> *>(tensor.data()));
//     return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
// }
    int A[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    Tensor a = make_tensor(&A[0], make_shape(Int<4>{}, Int<6>{}), make_stride(Int<6>{}, Int<1>{}));
    print(a);
    print_tensor(a);
    
    constexpr int num_element = decltype(size(a))::value;
    print(num_element);
    cutlass::NumericArrayConverter<float, int, num_element> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<int, num_element> *>(a.data()));
    Tensor b = make_tensor(a.data(), a.layout());

    print(b);
    print_tensor(b);
}