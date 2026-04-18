import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cuda.bindings.driver as cuda
import numpy as np
from cutlass.base_dsl.runtime import cuda as cuda_runtime
from cutlass.cute.runtime import make_ptr, nullptr


class TmaRoundTrip:
    def __init__(self, rows: int = 4, cols: int = 4):
        self.rows = rows
        self.cols = cols
        self.dtype = cutlass.Float32
        self.shape = (rows, cols)
        self.tile_shape = (rows, cols)
        self.layout = cute.make_layout(self.shape, stride=(cols, 1))
        self.threads_per_cta = 32

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        g_input = cute.make_tensor(input_ptr, self.layout)
        g_output = cute.make_tensor(output_ptr, self.layout)

        tma_load_atom, tma_input = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            g_input,
            self.layout,
            self.tile_shape,
        )
        tma_store_atom, tma_output = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            g_output,
            self.layout,
            self.tile_shape,
        )

        @cute.struct
        class SharedStorage:
            tma_barrier: cute.struct.MemRange[cutlass.Int64, 1]
            tile: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(self.layout)],
                1024,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_load_atom,
            tma_input,
            tma_store_atom,
            tma_output,
        ).launch(
            grid=[1, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_load_atom: cute.CopyAtom,
        m_input: cute.Tensor,
        tma_store_atom: cute.CopyAtom,
        m_output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_load_atom)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_store_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        s_tile = storage.tile.get_tensor(self.layout)
        tma_barrier = storage.tma_barrier.data_ptr()

        if tidx == 0:
            cute.arch.mbarrier_init(tma_barrier, 1)
        cute.arch.mbarrier_init_fence()
        pipeline.sync()

        s_tile_tma = cute.group_modes(s_tile, 0, 2)
        g_input_tiled = cute.zipped_divide(m_input, self.tile_shape)
        g_output_tiled = cute.zipped_divide(m_output, self.tile_shape)

        t_smem_load, t_gmem_load = cute.nvgpu.cpasync.tma_partition(
            tma_load_atom,
            0,
            cute.make_layout(1),
            s_tile_tma,
            g_input_tiled,
        )
        t_smem_store, t_gmem_store = cute.nvgpu.cpasync.tma_partition(
            tma_store_atom,
            0,
            cute.make_layout(1),
            s_tile_tma,
            g_output_tiled,
        )

        if tidx == 0:
            cute.copy(
                tma_load_atom,
                t_gmem_load[(None, 0)],
                t_smem_load,
                tma_bar_ptr=tma_barrier,
            )

        cute.arch.mbarrier_wait(tma_barrier, 0)
        pipeline.sync()

        cute.arch.fence_proxy("async.shared", space="cta")

        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        )

        if tidx == 0:
            cute.copy(
                tma_store_atom,
                t_smem_store,
                t_gmem_store[(None, 0)],
            )
            store_pipeline.producer_commit()
            store_pipeline.producer_acquire()
            store_pipeline.producer_tail()


def _device_ptr(ptr_value: int) -> cute.Pointer:
    return make_ptr(
        cutlass.Float32,
        ptr_value,
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )


def run_demo(rows: int = 4, cols: int = 4) -> None:
    cuda_runtime.initialize_cuda_context()
    stream = cuda_runtime.default_stream()

    host_input = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    host_output = np.zeros_like(host_input)

    num_bytes = host_input.nbytes
    device_input = cuda_runtime.allocate(num_bytes)
    device_output = cuda_runtime.allocate(num_bytes)

    try:
        cuda_runtime.memcpy_h2d(host_input.ctypes.data, device_input, num_bytes, stream)

        demo = TmaRoundTrip(rows=rows, cols=cols)
        compiled = cute.compile(
            demo,
            nullptr(cutlass.Float32, cutlass.AddressSpace.gmem, assumed_align=16),
            nullptr(cutlass.Float32, cutlass.AddressSpace.gmem, assumed_align=16),
            stream,
        )
        compiled(_device_ptr(int(device_input)), _device_ptr(int(device_output)), stream)
        cuda.cuStreamSynchronize(stream)

        cuda_runtime.memcpy_d2h(host_output.ctypes.data, device_output, num_bytes, stream)
        cuda.cuStreamSynchronize(stream)

        print("input:")
        print(host_input)
        print("output:")
        print(host_output)
        print("match:", np.array_equal(host_input, host_output))
    finally:
        cuda_runtime.deallocate(device_input)
        cuda_runtime.deallocate(device_output)


if __name__ == "__main__":
    run_demo()
