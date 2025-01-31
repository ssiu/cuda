from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="my_cuda_extension",
    ext_modules=[
        CUDAExtension(
            "my_cuda_extension",
            ["my_extension.cpp", "my_kernel.cu"]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)