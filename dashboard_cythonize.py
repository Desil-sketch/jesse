from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import multiprocessing
import os
import numpy 

#python3 dashboard_cythonize.py build_ext -j 8 --inplace 

os.environ['CFLAGS'] = '-O3 -march=native -Wall -mavx -flto=full -I /root/cpython-main/Include/Python.h'  #-emit-llvm -flto=full 
os.environ['CC'] = 'clang'
os.environ['LDSHARED'] = 'clang -shared'

Options.convert_range = True
Options.cache_builtins = True
List = [ 
    "jesse/exceptions/__init__.pyx",
    "jesse/helpers.pyx",
    "jesse/utils.pyx",
    "jesse/store/*.pyx",
    "jesse/research/*.pyx",
    "jesse/services/*.pyx",
    "jesse/strategies/*.pyx",
    "jesse/libs/dynamic_numpy_array/*.pyx",
    "jesse/libs/custom_json/*.pyx",
    "jesse/libs/*.pyx",
    "jesse/modes/*.pyx",
    "jesse/modes/optimize_mode/*.pyx",
    "jesse/enums/*.pyx",
    "jesse/models/*.pyx",
    "jesse/routes/*.pyx",
    "jesse/exchanges/*.pyx",
    "jesse/exchanges/sandbox/*.pyx",
]
exclusions = ["jesse/__init__.py"] 

setup(
    ext_modules=cythonize( List,nthreads=8,force=False,exclude=exclusions,compiler_directives={'language_level':3,'profile':False,'linetrace':False,'infer_types':True,'nonecheck':False,'optimize.use_switch':True,'optimize.unpack_method_calls':True,'initializedcheck':False,'cdivision':True,'wraparound':False,'boundscheck':False,}), include_dirs=[numpy.get_include()]
)
