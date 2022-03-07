
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

## Note about Transition

mlir-graphblas is transitioning away from lowering code targeting the SCF dialect and towards lowering code targeting
`linalg.generic`. This process is happening in tandem with changes to the sparse-tensor dialect's lowering of
`linalg.generic` with dynamically-shaped output. As a result, mlir-graphblas is temporarily pointing at the `mlir-ac`
conda package which is built from a branch off the LLVM project code. Once these changes are merged into the main branch
on LLVM, mlir-graphblas will be updated to target that. Until then, we will be out of sync with the LLVM project.