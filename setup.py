"""A setuptools based setup module.
"""

import sys
from setuptools import find_packages

REQUIREMENTS = [
    'numpy',
    'matplotlib',
    'scipy',
    'torch'
]

SETUP_KWARGS = dict(
    name='stereomatch',
    version="1.0.0",
    author='Otavio Gomes',
    author_email='otavio.b.gomes@gmail.com',
    description='Sample implementation of stereo matching algorithms',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5'
    ],
    packages=find_packages(exclude=['*._test']),
    install_requires=REQUIREMENTS,
    entry_points={
        'console_scripts': ['stm-run-pair=stereomatch.single_image_app:main']
    }
)


try:
    from skbuild import setup

    setup(**SETUP_KWARGS)
except ImportError:
    print('Note: scikit-build is required for developers source builds.',
          file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    print('Using production build')

    # pylint: disable=ungrouped-imports
    from setuptools import setup
    import torch
    from torch.utils.cpp_extension import (
        BuildExtension, CUDAExtension, include_paths)

    if torch.version.cuda is None:
        print("Your Torch version must support CUDA")
        sys.exit(1)

    SETUP_KWARGS.update(
        dict(ext_modules=[
            CUDAExtension('_cstereomatch', [
                'src/cuda_utils.cpp',
                'src/cuda_texture.cpp',
                'src/cuda_texture_gpu.cu',
                'src/cost.cpp',
                'src/ssd.cu',
                'src/birchfield_cost.cu',
                'src/winners_take_all.cu',
                'src/dynamic_programming.cu',
                'src/disparity_reduce.cpp',
                'src/semiglobal.cpp',
                'src/semiglobal_gpu.cu',
                'src/aggregation.cpp',
                'src/_cstereomatch.cpp'
            ],
                include_dirs=include_paths(
                    cuda=True) + ["include", "include/stereomatch"],
                extra_compile_args=["-std=c++17"]
            )
        ],
            cmdclass={'build_ext': BuildExtension}
        ))

    setup(**SETUP_KWARGS)
