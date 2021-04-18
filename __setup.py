from setuptools import setup
from Cython.Build import cythonize
from numpy import get_include

setup(
    name='vis_precision',
    ext_modules=cythonize('vis_precision.pyx'),
    include_dirs=[get_include()],
    zip_safe=False,
)
