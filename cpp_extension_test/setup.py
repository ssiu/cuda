from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

from cpp_extension.setup import nvcc_flags

cutlass_include_dirs = ["/cutlass/include", "/cutlass/tools/util/include"]

nvcc_flags = ["-std=c++17",
              "--expt-relaxed-constexpr",
              "-arch=sm_75",
              "-O3"]

setup(
    name="my_cuda_extension",
    ext_modules=[
        CUDAExtension(
            name="my_cuda_extension",
            sources=["my_extension.cpp", "my_kernel.cu"],
            include_dirs=cutlass_include_dirs,
            extra_compile_args={'nvcc': nvcc_flags}
        )
    ],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension}
)