from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
import versioneer

directive_defaults = get_directive_defaults()
directive_defaults["binding"] = True
directive_defaults["language_level"] = 3

setup(
    name="mlir-graphblas",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="MLIR dialect for GraphBLAS",
    author="Anaconda, Inc.",
    packages=find_packages(include=["mlir_graphblas", "mlir_graphblas.*"]),
    ext_modules=cythonize(
        Extension(
            "mlir_graphblas.wrap",
            language='c++',
            sources=["mlir_graphblas/wrap.pyx"],
            extra_compile_args=["-std=c++11"],
        )
    ),
    package_data={"mlir_graphblas": ["*.pyx"]},
    include_package_data=True,
    install_requires=["pymlir", "pygments"],
)
