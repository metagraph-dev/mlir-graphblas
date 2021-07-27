import sys
import pytest
import numpy as np

from io import StringIO
from contextlib import contextmanager
from typing import Generator, List

from mlir_graphblas import MlirJitEngine
from .jit_engine_test_utils import STANDARD_PASSES

###########
# Helpers #
###########


@contextmanager
def captured_stdout() -> Generator[List[str], None, None]:
    string_container: List[str] = []
    original_std_out = sys.stdout
    temporary_std_out = StringIO()
    sys.stdout = temporary_std_out
    yield string_container
    sys.stdout = original_std_out
    printed_output: str = temporary_std_out.getvalue()
    string_container.append(printed_output)
    return


#########
# Tests #
#########


def test_profile_single_return(
    engine: MlirJitEngine,
):
    mlir_text = """
func @single_value_slow_mul(%input_val: i32) -> i32 {
  %ci0 = constant 0 : i32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %num_iterations = constant 99 : index
  %answer = scf.for %i = %c0 to %num_iterations step %c1 iter_args(%current_sum=%ci0) -> (i32) {
    %new_sum = addi %current_sum, %input_val : i32
    scf.yield %new_sum : i32
  }
  return %answer : i32
}
"""
    try:
        engine.add(mlir_text, STANDARD_PASSES, profile=True)
    except NotImplementedError as e:
        if not sys.platform.startswith("linux"):
            pytest.skip(f"Profiling not supported on {sys.platform}.")
    compiled_func = getattr(engine, "single_value_slow_mul")
    input_val = 12

    with captured_stdout() as stdout_string_container:
        result = compiled_func(input_val)

    assert result == input_val * 99

    (stdout_string,) = stdout_string_container
    assert "Overhead" in stdout_string
    assert "Command" in stdout_string
    assert "Shared Object" in stdout_string
    assert "Symbol" in stdout_string
    assert "python3" in stdout_string

    return


def test_profile_multiple_returns(
    engine: MlirJitEngine,
):
    mlir_text = """
func @multiple_value_slow_mul(%input_val: i32) -> (i32, i32) {
  %ci0 = constant 0 : i32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %num_iterations = constant 99 : index
  %answer = scf.for %i = %c0 to %num_iterations step %c1 iter_args(%current_sum=%ci0) -> (i32) {
    %new_sum = addi %current_sum, %input_val : i32
    scf.yield %new_sum : i32
  }
  return %input_val, %answer : i32, i32
}
"""
    try:
        engine.add(mlir_text, STANDARD_PASSES, profile=True)
    except NotImplementedError as e:
        if not sys.platform.startswith("linux"):
            pytest.skip(f"Profiling not supported on {sys.platform}.")
    compiled_func = getattr(engine, "multiple_value_slow_mul")
    input_val = 12

    with captured_stdout() as stdout_string_container:
        redundant_input_val, result = compiled_func(input_val)

    assert redundant_input_val == input_val
    assert result == input_val * 99

    (stdout_string,) = stdout_string_container
    assert "Overhead" in stdout_string
    assert "Command" in stdout_string
    assert "Shared Object" in stdout_string
    assert "Symbol" in stdout_string
    assert "python3" in stdout_string
    return


@pytest.fixture(scope="module")
def engine():
    jit_engine = MlirJitEngine()
    return jit_engine
