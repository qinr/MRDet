# compile and debug rpsroi-align
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rpsroi_align_cuda',
    ext_modules=[
        CUDAExtension(
            name='rpsroi_align_cuda',
            sources=['src/rpsroi_align_cuda.cpp','src/rpsroi_align_kernel.cu',],
            extra_compile_args={
                'cxx': [],
                'nvcc': [
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension})
