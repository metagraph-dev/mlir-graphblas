# C++ Development Cheat Sheet

This document assumes you are only worried about building the C++
applications, not the Python library.

## Creating a new environment

```
conda create -n mgcpp -c metagraph/label/dev python=3.8 cmake ninja mlir
```

You will also likely want to have a copy of LLVM sitting around somewhere you can browse and search:
```
git clone https://github.com/llvm/llvm-project/
```

## Setting up VSCode

VSCode's Intellisense features are basically mandatory to make progress, especially given how incomplete MLIR's documentation is.  At the top level of the repository, create a file `.vscode/c_cpp_properties.json` which contains:

```
{
    "configurations": [
        {
            "name": "macOS",
            "includePath": [
                "~/miniconda3/envs/mgcpp/include"
            ],
            "macFrameworkPath": [
                "/System/Library/Frameworks",
                "/Library/Frameworks"
            ],
            "intelliSenseMode": "macos-clang-x64",
            "compilerPath": "/usr/bin/g++",
            "cStandard": "c11",
            "cppStandard": "c++14"
        }
    ],
    "version": 4
}
```

Adjust `includePath` to match your environment.  The `name` attribute doesn't appear to have any special meaning.

## Setting up CMake build

From the top level of the repository

```
conda activate mgcpp
cd mlir_graphblas/src/
mkdir build
cd build
cmake -G Ninja .. -DMLIR_DIR=$CONDA_PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_BUILD_TYPE=Debug
```
This command line overrides cmake to force it to use the system g++ (which is actually clang) rather than any version of clang that might be in your conda environment.

## Build and Test Loop

Next, you'll likely run the following command over and over when testing lowering:
```
cmake --build . -v && bin/lowering-test
```

If you have a crash, the backtrace is not very useful, so you'll want to run the `lowering-test` in lldb:
```
lldb bin/lowering-test
```
