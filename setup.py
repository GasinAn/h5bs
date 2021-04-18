from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include

setup(
    name='dnb_float32',
    ext_modules=cythonize('dnb_float32.pyx'),
    include_dirs=[get_include()],
    zip_safe=False,
)
