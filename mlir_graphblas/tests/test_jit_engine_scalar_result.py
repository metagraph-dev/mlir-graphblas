
from .jit_engine_test_utils import MLIR_TYPE_TO_NP_TYPE, STANDARD_PASSES

import mlir_graphblas
import pytest
import numpy as np

SCALAR_RESULT_TEST_CASES = [ # elements are ( mlir_template, args, expected_result)
    
    ( # arbitrary size 1D tensor -> scalar
        """
func @{func_name}(%arg0: tensor<?x{mlir_type}>) -> {mlir_type} {{
  %c3 = constant 3 : index
  %ans = tensor.extract %arg0[%c3] : tensor<?x{mlir_type}>
  return %ans : {mlir_type}
}}
""",
        [
            np.arange(8)
        ],
        3
    ),
    
    ( # arbitrary size 1D tensor, scalar -> scalar
        """
func @{func_name}(%tensor_arg: tensor<?x{mlir_type}>, %scalar_arg: {mlir_type}) -> {mlir_type} {{
  %c3 = constant 3 : index
  %element_3 = tensor.extract %tensor_arg[%c3] : tensor<?x{mlir_type}>
  %ans = {linalg_mul} %element_3, %scalar_arg : {mlir_type}
  return %ans : {mlir_type}
}}
""",
        [
            np.arange(8),
            10
        ],
        30
    ),
    
    ( # scalar, arbitrary size 1D tensor -> scalar
        """
func @{func_name}(%scalar_arg: {mlir_type}, %tensor_arg: tensor<?x{mlir_type}>) -> {mlir_type} {{
  %c3 = constant 3 : index
  %element_3 = tensor.extract %tensor_arg[%c3] : tensor<?x{mlir_type}>
  %ans = {linalg_mul} %element_3, %scalar_arg : {mlir_type}
  return %ans : {mlir_type}
}}
""",
        [
            10,
            np.arange(8)
        ],
        30
    ),
    
    ( # arbitrary size 1D tensor, arbitrary size 1D tensor -> scalar
        """
func @{func_name}(%arg0: tensor<?x{mlir_type}>, %arg1: tensor<?x{mlir_type}>) -> {mlir_type} {{
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %arg0_3 = tensor.extract %arg0[%c3] : tensor<?x{mlir_type}>
  %arg1_4 = tensor.extract %arg1[%c4] : tensor<?x{mlir_type}>
  %ans = {linalg_mul} %arg0_3, %arg1_4 : {mlir_type}
  return %ans : {mlir_type}
}}
""",
        [
            np.arange(8),
            np.arange(5),
        ],
        12
    ),
    
    ( # scalar, arbitrary size 1D tensor, arbitrary size 1D tensor -> scalar
        """
func @{func_name}(%scalar: {mlir_type}, %arg0: tensor<?x{mlir_type}>, %arg1: tensor<?x{mlir_type}>) -> {mlir_type} {{
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %arg0_3 = tensor.extract %arg0[%c3] : tensor<?x{mlir_type}>
  %arg1_4 = tensor.extract %arg1[%c4] : tensor<?x{mlir_type}>
  %tensor_element_result = {linalg_mul} %arg0_3, %arg1_4 : {mlir_type}
  %ans = {linalg_mul} %tensor_element_result, %scalar : {mlir_type}
  return %ans : {mlir_type}
}}
""",
        [
            2,
            np.arange(8),
            np.arange(5),
        ],
        24
    ),
]

def test_jit_engine_scalar_result():
    
    engine = mlir_graphblas.MlirJitEngine()
    for test_case_index, test_case in enumerate(SCALAR_RESULT_TEST_CASES):
        for mlir_type, np_type in MLIR_TYPE_TO_NP_TYPE.items():
            func_name = f"func_{test_case_index}_{mlir_type}"

            if issubclass(np_type, np.integer):
                linalg_mul = 'muli'
            elif issubclass(np_type, np.floating):
                linalg_mul = 'mulf'
            else:
                raise ValueError(f"No MLIR type for {np_type}.")
            
            mlir_template, args, expected_result = test_case
            mlir_text = mlir_template.format(
                func_name=func_name,
                linalg_mul=linalg_mul,
                mlir_type=mlir_type,
            )
            args = [arg.astype(np_type) if isinstance(arg, np.ndarray) else arg for arg in args]
            expected_result = expected_result.astype(np_type) if isinstance(expected_result, np.ndarray) else expected_result
            
            engine.add(mlir_text, STANDARD_PASSES)
            
            compiled_func = engine[func_name]
            result = compiled_func(*args)
            assert np.all(result == expected_result), f"""
Input MLIR: 
{mlir_text}

Inputs: {args}
Result: {result}
Expected Result: {expected_result}
"""
