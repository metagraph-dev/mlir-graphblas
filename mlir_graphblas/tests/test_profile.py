import sys
import os
import pytest
import uuid
import tempfile
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
    num_iterations = 12345678
    function_name = f"single_value_slow_mul_{int(uuid.uuid4())}"
    mlir_text = f"""
func @{function_name}(%input_val: i64) -> i64 {{
  %ci0 = constant 0 : i64
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %num_iterations = constant {num_iterations} : index
  %answer = scf.for %i = %c0 to %num_iterations step %c1 iter_args(%current_sum=%ci0) -> (i64) {{
    %new_sum = addi %current_sum, %input_val : i64
    scf.yield %new_sum : i64
  }}
  return %answer : i64
}}
"""
    try:
        engine.add(mlir_text, STANDARD_PASSES, profile=True)
    except NotImplementedError as e:
        if not sys.platform.startswith("linux"):
            pytest.skip(f"Profiling not supported on {sys.platform}.")
    except RuntimeError as e:
        (err_string,) = e.args
        if err_string == (
            "Profiling not permitted since the contents of "
            "/proc/sys/kernel/perf_event_paranoid must be -1."
        ):
            pytest.skip(f"Profiling not permitted.")
        else:
            raise e
    compiled_func = getattr(engine, function_name)
    input_val = 12

    with captured_stdout() as stdout_string_container:
        result = compiled_func(input_val)

    assert result == input_val * num_iterations

    (stdout_string,) = stdout_string_container
    assert "Source code & Disassembly of" in stdout_string
    assert "Sorted summary for file" in stdout_string
    assert "Percent" in stdout_string
    return


def test_profile_multiple_returns(
    engine: MlirJitEngine,
):
    result_dir = tempfile.TemporaryDirectory()
    num_iterations = 12345678
    function_name = f"multivalue_slow_mul_{int(uuid.uuid4())}"
    mlir_text = f"""
func @{function_name}(%input_val: i64) -> (i64, i64) {{
  %ci0 = constant 0 : i64
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %num_iterations = constant {num_iterations} : index
  %answer = scf.for %i = %c0 to %num_iterations step %c1 iter_args(%current_sum=%ci0) -> (i64) {{
    %new_sum = addi %current_sum, %input_val : i64
    scf.yield %new_sum : i64
  }}
  return %input_val, %answer : i64, i64
}}
"""
    try:
        engine.add(
            mlir_text,
            STANDARD_PASSES,
            profile=True,
            profile_result_directory=result_dir.name,
        )
    except NotImplementedError as e:
        if not sys.platform.startswith("linux"):
            pytest.skip(f"Profiling not supported on {sys.platform}.")
    except RuntimeError as e:
        (err_string,) = e.args
        if err_string == (
            "Profiling not permitted since the contents of "
            "/proc/sys/kernel/perf_event_paranoid must be -1."
        ):
            pytest.skip(f"Profiling not permitted.")
        else:
            raise e
    compiled_func = getattr(engine, function_name)
    input_val = 12

    with captured_stdout() as stdout_string_container:
        redundant_input_val, result = compiled_func(input_val)

    assert redundant_input_val == input_val
    assert result == input_val * num_iterations

    (stdout_string,) = stdout_string_container
    assert "Source code & Disassembly of" in stdout_string
    assert "Sorted summary for file" in stdout_string
    assert "Percent" in stdout_string

    assert (
        len(
            [
                file_name
                for file_name in os.listdir(result_dir.name)
                if file_name.startswith("perf-") and file_name.endswith(".data")
            ]
        )
        == 1
    )
    return


@pytest.fixture(scope="module")
def engine():
    jit_engine = MlirJitEngine()
    return jit_engine
