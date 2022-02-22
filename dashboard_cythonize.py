from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
import multiprocessing
import os
import numpy 

    # @cython.wraparound(True)
    #cython:wraparound=False
os.environ['CFLAGS'] = '-O3 -march=native -Wall -mavx -flto=full -I /root/cpython-main/Include/Python.h'  #-emit-llvm -flto=full 
os.environ['CC'] = 'clang'
os.environ['LDSHARED'] = 'clang -shared'

#python3 dashboard_cythonize.py build_ext -j 8 --inplace 
 
Options.annotate = True
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
    # "//mnt/c/Python39/Algotrading/custom_indicators/cython_jma.pyx",
    # "cyt_xt1.pyx",
    # "jesse/__init__.py",
    # "//mnt/c/Python39/Algotrading/strategies/TestStrategy/*.pyx",
]
exclusions = ["jesse/__init__.py","jesse/modes/optimize_mode/__init__.pyx","jesse/services/web.pyx",] 

# other compiler_directs={''profile':True,'linetrace':True, force=True}
# other setup option = language = 'c++',
# other environ = std=c++11 -wall -flto=thin
setup(
    ext_modules=cythonize( List,nthreads=8,force=False,exclude=exclusions,compiler_directives={'language_level':3,'profile':False,'linetrace':False,'infer_types':True,'nonecheck':False,'optimize.use_switch':True,'optimize.unpack_method_calls':True,'initializedcheck':False,'cdivision':True,'wraparound':False,'boundscheck':False,}), include_dirs=[numpy.get_include()]
)