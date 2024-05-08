try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    "mcubes",
    sources=[
        "src/mcubes.pyx",
        "src/pywrapper.cpp",
        "src/marchingcubes.cpp",
    ],
    language="c++",
    extra_compile_args=["-std=c++11"],
    include_dirs=[numpy_include_dir],
)


# Gather all extension modules
ext_modules = [
    mcubes_module,
]

setup(name = "mcubes", ext_modules=cythonize(ext_modules), cmdclass={"build_ext": BuildExtension})

