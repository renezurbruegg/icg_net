try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.extension import Extension

import numpy
from torch.utils.cpp_extension import BuildExtension

numpy_include_dir = numpy.get_include()

mcubes_module = Extension(
    "mcubes",
    sources=[
        "src/mcubes.cpp",
        "src/pywrapper.cpp",
        "src/marchingcubes.cpp",
    ],
    language="c++",
    extra_compile_args=["-std=c++11"],
    include_dirs=[numpy_include_dir],
)

setup(name="mcubes", ext_modules=[mcubes_module], cmdclass={"build_ext": BuildExtension})

