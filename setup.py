import os
import sys
import subprocess
import distutils
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
import versioneer

if sys.platform == "darwin":
    # Create a dynamiclib instead of bundles so that it may be linked.
    # https://stackoverflow.com/questions/32419594/how-to-create-a-dylib-c-extension-on-mac-os-x-with-distutils-and-or-setuptools/32765319#32765319
    from distutils import sysconfig

    vars = sysconfig.get_config_vars()
    vars["LDSHARED"] = vars["LDSHARED"].replace("-bundle", "-dynamiclib")

##################################
# SparseTensorUtils.cpp Cython Wrapper #
##################################

directive_defaults = get_directive_defaults()
directive_defaults["binding"] = True
directive_defaults["language_level"] = 3

environment_include_dir = os.path.join(sys.exec_prefix, "include")
include_dirs = [np.get_include(), environment_include_dir]
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
annotate = True  # Creates html file

#########
# setup #
#########

ext_modules = cythonize(
    [
        Extension(
            "mlir_graphblas.sparse_utils",
            language="c++",
            sources=["mlir_graphblas/sparse_utils.pyx"],
            extra_compile_args=["-std=c++11"],
            include_dirs=include_dirs,
            define_macros=define_macros,
        ),
        Extension(
            "mlir_graphblas.random_utils",
            language="c++",
            sources=["mlir_graphblas/random_utils.pyx"],
            extra_compile_args=["-std=c++11"],
            include_dirs=include_dirs,
            define_macros=define_macros,
        ),
    ],
    annotate=annotate,
)

ext_modules.append(
    Extension(
        "mlir_graphblas.SparseTensorUtils",
        sources=["mlir_graphblas/SparseTensorUtils.cpp"],
        include_dirs=[environment_include_dir],
        extra_compile_args=["-std=c++11"],
    )
)

ext_modules.append(
    Extension(
        "mlir_graphblas.RandomUtils",
        sources=["mlir_graphblas/RandomUtils.cpp"],
        include_dirs=[environment_include_dir],
        extra_compile_args=["-std=c++11"],
    )
)

setup(
    name="mlir-graphblas",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="MLIR dialect for GraphBLAS",
    author="Anaconda, Inc.",
    packages=find_packages(include=["mlir_graphblas", "mlir_graphblas.*"]),
    ext_modules=ext_modules,
    package_data={"mlir_graphblas": ["*.pyx"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "tersify_mlir=mlir_graphblas.tools:tersify_mlir_cli",
        ]
    },
    install_requires=["pymlir", "pygments"],
)
