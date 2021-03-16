import os
import subprocess
import distutils
import numpy as np
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Compiler.Options import get_directive_defaults
import versioneer

##################################
# SparseUtils.cpp Cython Wrapper #
##################################

directive_defaults = get_directive_defaults()
directive_defaults["binding"] = True
directive_defaults["language_level"] = 3

include_dirs = [np.get_include()]
if "CONDA_PREFIX" in os.environ:
    conda_include_dir = os.environ["CONDA_PREFIX"]
    conda_include_dir = os.path.join(conda_include_dir, "include")
    include_dirs.append(conda_include_dir)
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
annotate = True  # Creates html file

#################################
# SparseUtils.cpp Shared Object #
#################################


class CompileSparseUtilsSharedObjectCommand(distutils.cmd.Command):
    """
    Example Usage:
        python3 setup.py compile_sparse_utils_so
    """

    sparse_utils_cpp_location = "./mlir_graphblas/SparseUtils.cpp"
    description = f"Compile {sparse_utils_cpp_location}."
    user_options = []

    def initialize_options(self) -> None:
        return

    def finalize_options(self) -> None:
        return

    def run(self) -> None:

        self.announce(
            f"Compiling {self.sparse_utils_cpp_location}.", level=distutils.log.INFO
        )
        include_options = (
            f"-I{os.environ['CONDA_PREFIX']}/include"
            if "CONDA_PREFIX" in os.environ
            else ""
        )
        compile_command = f"g++ -c -Wall -Werror -fpic {include_options} {self.sparse_utils_cpp_location} && g++ -shared -o ./mlir_graphblas/SparseUtils.so SparseUtils.o"
        process = subprocess.Popen(
            "/bin/bash",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(compile_command.encode())
        if process.returncode != 0:
            raise RuntimeError(
                f"""
Could not compile {self.sparse_utils_cpp_location}.

Compile Command:

{compile_command}

STDOUT:

{stdout.decode()}

STDERR:

{stderr.decode()}

"""
            )
        self.announce(
            f"Finished compiling {self.sparse_utils_cpp_location}.",
            level=distutils.log.INFO,
        )
        return


#########
# setup #
#########

cmdclass = {
    "compile_sparse_utils_so": CompileSparseUtilsSharedObjectCommand,
}
cmdclass.update(versioneer.get_cmdclass())

setup(
    name="mlir-graphblas",
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    description="MLIR dialect for GraphBLAS",
    author="Anaconda, Inc.",
    packages=find_packages(include=["mlir_graphblas", "mlir_graphblas.*"]),
    ext_modules=cythonize(
        Extension(
            "mlir_graphblas.wrap",
            language="c++",
            sources=["mlir_graphblas/wrap.pyx"],
            extra_compile_args=["-std=c++11"],
            include_dirs=include_dirs,
            define_macros=define_macros,
        ),
        annotate=annotate,
    ),
    package_data={"mlir_graphblas": ["*.pyx"]},
    include_package_data=True,
    install_requires=["pymlir", "pygments"],
)
