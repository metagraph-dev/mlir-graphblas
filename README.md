
# MLIR dialect for GraphBLAS + Python tools

[![Conda Version](https://img.shields.io/conda/v/metagraph/mlir-graphblas.svg)](https://anaconda.org/metagraph/mlir-graphblas)
[![Build Status](https://github.com/metagraph-dev/mlir-graphblas/actions/workflows/test_and_deploy.yml/badge.svg?branch=main)](https://github.com/metagraph-dev/mlir-graphblas/actions/workflows/test_and_deploy.yml?query=branch%3Amain)

*Note that this code currently requires [llvm-project@bd0cae6](https://github.com/llvm/llvm-project/commit/bd0cae6).*

## graphblas-opt

In order to build `graphblas-opt`, run `python3 build.py` from `mlir_graphblas/src/`. This will build `graphblas-opt` and run the tests for `graphblas-opt`. The built files will be stored in `mlir_graphblas/src/build`.

`build.py` does not rebuild from scratch by default. To perform a clean build, run  `python3 build.py -build-clean`. 

## Linting with clang-format

Ensure that `clang-format` is available (install the `clang-tools` conda package) and run:

```
./run-clang-format.py -r mlir_graphblas/src/
```

If changes required, can make changes in place with
```
./run-clang-format.py -i -r mlir_graphblas/src/
```
