from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

#from cpp_extension.setup import nvcc_flags

cutlass_include_dirs = ["/root/cuda/cpp_extension_test/cutlass/include", "/root/cuda/cpp_extension_test/cutlass/tools/util/include"]

nvcc_flags = ["-std=c++17",
              "--expt-relaxed-constexpr",
              "-arch=sm_75",
              "-O3"]

setup(
    name="flash_attn_turing",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_turing",
            sources=["flash_attn_turing.cpp", "flash_fwd_v0.cu"],
            include_dirs=cutlass_include_dirs,
            extra_compile_args={'nvcc': nvcc_flags}
        )
    ],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension}
)