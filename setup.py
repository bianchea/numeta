import numpy
from setuptools import Extension, find_packages, setup

module = Extension(
    "numeta._signature",
    sources=["numeta/_signature.c"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=["-O3"],
    optional=True,
)

setup(
    name="numeta",
    packages=find_packages(),
    ext_modules=[module],
)
