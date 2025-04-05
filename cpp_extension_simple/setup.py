from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

#from cpp_extension.setup import nvcc_flags

nvcc_flags = ["-std=c++17",
              "--expt-relaxed-constexpr",
              "-arch=sm_75",
              "-O3"]

setup(
    name="simple",
    ext_modules=[
        CUDAExtension(
            name="simple",
            sources=["simple.cpp"],
            extra_compile_args={'nvcc': nvcc_flags}
        )
    ],
    install_requires=["torch"],
    cmdclass={"build_ext": BuildExtension}
)