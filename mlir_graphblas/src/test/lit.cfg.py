# -*- Python -*-

import glob
import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "GRAPHBLAS"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.graphblas_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))


_SCRIPT_DIR = os.path.dirname(__file__)
_BUILD_DIR = os.path.join(_SCRIPT_DIR, "..", "..")
_BUILD_DIR = os.path.abspath(_BUILD_DIR)
_SPARSE_UTILS_SO_PATTERN = os.path.join(_BUILD_DIR, "SparseUtils*.so")
[_SPARSE_UTILS_SO] = glob.glob(_SPARSE_UTILS_SO_PATTERN)
config.substitutions.append(
    (
        "%sparse_utils_so",
        _SPARSE_UTILS_SO,
    )
)

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "Examples", "CMakeLists.txt", "README.txt", "LICENSE.txt"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.graphblas_obj_root, "test")
config.graphblas_tools_dir = os.path.join(config.graphblas_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.graphblas_tools_dir, config.llvm_tools_dir]
tools = ["graphblas-opt"]

llvm_config.add_tool_substitutions(tools, tool_dirs)
