from setuptools import setup, find_packages
import versioneer

setup(
    name="mlir-graphblas",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="MLIR dialect for GraphBLAS",
    author="Anaconda, Inc.",
    packages=find_packages(include=["mlir_graphblas", "mlir_graphblas.*"]),
    include_package_data=True,
    install_requires=["pymlir", "pygments"],
)
