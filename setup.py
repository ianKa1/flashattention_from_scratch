from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="flash_attn_ext",
    ext_modules=[
        CUDAExtension(
            name="flash_attn_ext",
            sources=["flash_attn_cuda.cu", "flash_attn.cpp"],
            extra_cuda_cflags=["--use_fast_math"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)