#include <cute/tensor.hpp>

using namespace cute;

int main()
{
#if 1
  {

//    Copy_Atom<UniversalCopy<double>, double> copy_atom;

    auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                                         Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                         Layout<Shape< _2,_1>>{});
//    auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
//                                      Layout<Shape<_4,_8>, Stride<_1,_4>>{},
//                                      Layout<Shape< _4,_1>>{});
//    auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
//                                      Layout<Shape<_16,_8>>{},
//                                      Layout<Shape< _1,_2>>{});
//     auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, float>{},
//                                       Layout<Shape<_32,_8>>{},  // 32x1 threads
//                                       Layout<Shape< _1,_8>>{}); //  1x4 values

    print_latex(tiled_copy);
  }
#endif


#if 0
  {
//    Copy_Atom<UniversalCopy<double>, double> copy_atom;
//
////    auto tiled_copy = make_tiled_copy(copy_atom,
////                                      Layout<Shape<Shape<_32,_1>,_1>, Stride<Stride<_2,_1>,_1>>{},  // 32x1 threads
////                                      Layout<Shape< _1,_1>>{}); //  1x4 values
//    //const int
////    auto thr_copy = tiled_copy.get_thread_slice(0);
////    print_layout(thr_copy);
//   // print_latex(tiled_copy);
//
//
////    auto layout_2x4 = make_layout(make_shape (2, make_shape (2,2)),
////                              make_stride(4, make_stride(2,1)));
////    auto layout_2x4 = make_layout(make_shape (1, make_shape (2,2)),
////                              make_stride(4, make_stride(2,1)));
//    auto layout_2x4 = make
//    _layout(make_shape ( make_shape (2,2), 1),
//                              make_stride(make_stride(2,1), 4));
//    auto flat_layout = flatten(layout_2x4);
//    print_layout(layout_2x4);
//    print_layout(flat_layout);
  }
#endif

#if 0
  {
    Copy_Atom<UniversalCopy<double>, double> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<Shape<_2, _2>, _2>, Stride<Stride<_4, _1>, _2>>{},  // 8x4 threads
                                      Layout<Shape< _1,_1>>{}); //  1x4 values

    print_latex(tiled_copy);
  }
#endif

#if 0
  {
    Copy_Atom<UniversalCopy<double>, double> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_32,_1>>{},  // 32x1 threads
                                      Layout<Shape< _1,_4>>{}); //  1x4 values

    print_latex(tiled_copy);
  }
#endif


#if 0
  {
    // The canonical LDSM_N image
    Copy_Atom<SM75_U32x1_LDSM_N, uint32_t> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_8,_4>, Stride<_4,_1>>{}); // 8x4 RowMajor threads

    print_latex(tiled_copy);
  }
#endif

#if 0
  {
    // The canonical LDSM_T image
    Copy_Atom<SM75_U16x2_LDSM_T, uint16_t> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_8,_4>, Stride<_4,_1>>{},  // 8x4 RowMajor threads
                                      Layout<Shape<_1,_2>>{});                // 1x2 values per thread

    print_latex(tiled_copy);
  }
#endif

#if 0
  {
    // Generate a TiledCopy layout from a TiledMMA
    SM80_16x8x16_F32F16F16F32_TN mma;
    auto tiled_mma = make_tiled_mma(mma);   // HMMA.16816 warp-wide MmaAtom. 32 threads, 1 warp. Each thread owns 4 output values.

    // Pick an Atom
    //Copy_Atom<DefaultCopy, uint16_t> copy_atom;
    //Copy_Atom<SM75_U16x2_LDSM_T, uint16_t> copy_atom;
    //Copy_Atom<SM75_U16x4_LDSM_T, uint16_t> copy_atom;
    Copy_Atom<SM75_U16x8_LDSM_T, uint16_t> copy_atom;

    // Define the layout of threads and values from the tiled_mma
    auto tiled_copy = make_tiled_copy_A(copy_atom, tiled_mma);

    print_latex(tiled_copy);
  }
#endif

#if 0
  {
    Copy_Atom<SM75_U16x2_LDSM_T, uint16_t> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                      Layout<Shape<_2,_1>>{});

    print_latex(tiled_copy);
  }
#endif

#if 0
  {
    Copy_Atom<SM75_U16x2_LDSM_N, uint16_t> copy_atom;

    auto tiled_copy = make_tiled_copy(copy_atom,
                                      Layout<Shape<_32,_1>>{},
                                      Layout<Shape< _1,_8>>{});

    print_latex(tiled_copy);
  }
#endif
}