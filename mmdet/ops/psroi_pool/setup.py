# compile and debug psroi-pool
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='psroi_pool_cuda',
    ext_modules=[
        CUDAExtension(
            name='psroi_pool_cuda',
            sources=['src/psroi_pool_cuda.cpp','src/psroi_pool_kernel.cu',],
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
