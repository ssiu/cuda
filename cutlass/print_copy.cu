#include <cute/tensor.hpp>
using namespace cute;

int main()
{




#if 0

    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 2 * kMmaEURepeatK * get<2>(mma_atom_shape{});
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    MMA tiled_mma;

    //auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{});

    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, half_t>;
    using S2RCopyAtomA = s2r_copy_atom;
    //using S2RCopyAtomB = s2r_copy_atom;

    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    print_latex(s2r_tiled_copy_a);

#endif


// https://github.com/andylolu2/simpleGEMM/blob/master/gemm_config_sm75.cuh
#if 1
{
    static constexpr int64_t BLK_M = 128;
    static constexpr int64_t BLK_N = 128;
    static constexpr int64_t BLK_K = 64;
    static constexpr int64_t NumThreads = 128;  // 4 warps
  
    static constexpr int AccessSizeBits = 128;
    static constexpr int ElemsPerLoad = AccessSizeBits / sizeof_bits_v<half_t>; // 8
    static constexpr int SmemAtomInner = 64;
    static constexpr int SmemAtomOuter = ElemsPerLoad; // 8
    static constexpr int ThreadsPerRow = SmemAtomInner / ElemsPerLoad; // 8
    
//     using BlockShapeA = Shape<Int<BLK_M>, Int<BLK_K>>;
//     using BlockShapeB = Shape<Int<BLK_N>, Int<BLK_K>>;
    using BlockShapeA = Shape<Int<8>, Int<BLK_K>>;
    using BlockShapeB = Shape<Int<8>, Int<BLK_K>>;

    using SmemLayoutAtom = decltype(composition(Swizzle<3, 3, 3>{},
                                                    Layout<
                                                        Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
                                                        Stride<Int<SmemAtomInner>, Int<1>>>{})); // (8,64)

   
    // Layout of each block of A/B in shared memory
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, BlockShapeA{}));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{}, BlockShapeB{}));

   
    // The copy atom for gmem -> smem (read A/B) or rmem -> gmem (store C).
    using GmemCopyAtom = Copy_Atom<
        AutoVectorizingCopyWithAssumedAlignment<AccessSizeBits>, half_t>;
    // The thread layout for one tile of the gmem -> smem copy.
    using GmemCopyThreadLayoutA = Layout<Shape<Int<NumThreads / ThreadsPerRow>, Int<ThreadsPerRow>>,
                                             Stride<Int<ThreadsPerRow>, Int<1>>>;
    // The value layout for each thread in the gmem -> smem copy.
    using GmemCopyValLayoutA = Layout<Shape<Int<1>, Int<ElemsPerLoad>>>;

   
    // Tiled copy of A/B from gmem -> smem
    using GmemCopyA = decltype(make_tiled_copy(GmemCopyAtom{},
                                                   GmemCopyThreadLayoutA{},
                                                   GmemCopyValLayoutA{}));
    using GmemCopyB = GmemCopyA;

   
    // The atom of the smem -> rmem copy for A/B. Loads 4 8x8 matrices (distributed across threads) at a time.
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
    // The atom for the MMA operation. Each atom is a warp-wise instruction that computes a 16x8x8 mma (with tensor cores).
    using MmaAtom = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    // We have 128 threads, so we use 4 warps laid out in 2x2x1.
    using MmaAtomLayout = Layout<Shape<Int<2>, Int<2>, Int<1>>>;
    // We want to use the `ldmatrix.x4.m8n8` instruction which loads 4 8x8 matrices for maximum efficiency.
    // To make the operands A and B divisible into 4 8x8 matrices, we expand the problem size for each warp to 16x16x16.
    // Accounting for the fact that we use 4 warps laid out in 2x2x1, the full tile size is 32x32x16.
    using MmaTiledShape = Tile<Int<32>, Int<32>, Int<16>>;
   
    // Tiled mma operation
    using TiledMMA = TiledMMA<MmaAtom, MmaAtomLayout, MmaTiledShape>;
    // Tiled copy of A from smem -> rmem
    using SmemCopyA = decltype(make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));
    // Tiled copy of B from smem -> rmem
    using SmemCopyB = decltype(make_tiled_copy_B(SmemCopyAtom{}, TiledMMA{}));


    GmemCopyA gmem_copy_a;
    SmemCopyA smem_copy_a;
    SmemCopyB smem_copy_b;
    SmemLayoutA smem_layout_a;
    SmemLayoutAtom smem_layout_atom;
    print_latex(smem_layout_atom);

    //print_latex(smem_layout_a);
    //print_latex(gmem_copy_a);
    //print_latex(smem_copy_a);
    //print_latex(smem_copy_b);


}
#endif






#if 0
  {
        using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
        // The atom for the MMA operation. Each atom is a warp-wise instruction that computes a 16x8x8 mma (with tensor cores).
        using MmaAtom = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
        // We have 128 threads, so we use 4 warps laid out in 2x2x1.
        using MmaAtomLayout = Layout<Shape<Int<1>, Int<1>, Int<1>>>;
        // We want to use the `ldmatrix.x4.m8n8` instruction which loads 4 8x8 matrices for maximum efficiency.
        // To make the operands A and B divisible into 4 8x8 matrices, we expand the problem size for each warp to 16x16x16.
        // Accounting for the fact that we use 4 warps laid out in 2x2x1, the full tile size is 32x32x16.
        using MmaTiledShape = Tile<Int<32>, Int<8>, Int<8>>;

        using TiledMMA = TiledMMA<MmaAtom, MmaAtomLayout, MmaTiledShape>;
        // Tiled copy of A from smem -> rmem
        using SmemCopyA = decltype(make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));

        SmemCopyA smem_copy_a;

        print_latex(smem_copy_a);

  }
#endif

#if 0
  {
        using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
        // The atom for the MMA operation. Each atom is a warp-wise instruction that computes a 16x8x8 mma (with tensor cores).
        using MmaAtom = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
        // We have 128 threads, so we use 4 warps laid out in 2x2x1.
        using MmaAtomLayout = Layout<Shape<Int<2>, Int<2>, Int<1>>>;
        // We want to use the `ldmatrix.x4.m8n8` instruction which loads 4 8x8 matrices for maximum efficiency.
        // To make the operands A and B divisible into 4 8x8 matrices, we expand the problem size for each warp to 16x16x16.
        // Accounting for the fact that we use 4 warps laid out in 2x2x1, the full tile size is 32x32x16.
        using MmaTiledShape = Tile<Int<32>, Int<32>, Int<16>>;

        using TiledMMA = TiledMMA<MmaAtom, MmaAtomLayout, MmaTiledShape>;
        // Tiled copy of A from smem -> rmem
        using SmemCopyA = decltype(make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));

        SmemCopyA smem_copy_a;

        print_latex(smem_copy_a);

  }
#endif

#if 0
  {

//    Copy_Atom<UniversalCopy<double>, double> copy_atom;

//     auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
//                                          Layout<Shape<_32,_8>, Stride<_1,_32>>{},
//                                          Layout<Shape< _4,_1>>{});
   auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                                     Layout<Shape<_128,_2>, Stride<_2,_1>>{},
                                     Layout<Shape< _1,_4>>{});
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

//     auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
//                                          Layout<Shape<_4,_8>, Stride<_1,_4>>{},
//                                          Layout<Shape< _2,_1>>{});
   auto tiled_copy = make_tiled_copy(Copy_Atom<DefaultCopy, half_t>{},
                                     Layout<Shape<_4,_8>, Stride<_1,_4>>{},
                                     Layout<Shape< _4,_1>>{});
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