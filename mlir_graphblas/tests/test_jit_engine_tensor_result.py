from .jit_engine_test_utils import MLIR_TYPE_TO_NP_TYPE, STANDARD_PASSES

import mlir_graphblas
import pytest
import numpy as np

TENSOR_RESULT_TEST_CASES = [  # elements are ( mlir_template, args, expected_result x)
    (  # fixed size tensor , fixed size tensor -> fixed size tensor
        """
#trait_add = {{
 indexing_maps = [
   affine_map<(i, j) -> (i, j)>,
   affine_map<(i, j) -> (i, j)>,
   affine_map<(i, j) -> (i, j)>
 ],
 iterator_types = [\"parallel\", \"parallel\"],
 doc = \" !!! DUMMY DOCUMENTATION !!! \"
}}
func @{func_name}(%arga: tensor<2x4x{mlir_type}>, %argb: tensor<2x4x{mlir_type}>) -> tensor<2x4x{mlir_type}> {{
 %answer = linalg.generic #trait_add
    ins(%arga, %argb: tensor<2x4x{mlir_type}>, tensor<2x4x{mlir_type}>)
   outs(%arga: tensor<2x4x{mlir_type}>) {{
     ^bb(%a: {mlir_type}, %b: {mlir_type}, %s: {mlir_type}):
       %sum = {linalg_add} %a, %b : {mlir_type}
       linalg.yield %sum : {mlir_type}
 }} -> tensor<2x4x{mlir_type}>
 return %answer : tensor<2x4x{mlir_type}>
}}
""",
        [np.arange(2 * 4).reshape((2, 4)), np.full([2, 4], 10),],
        np.array([[10, 11, 12, 13], [14, 15, 16, 17]]),
    ),
    (  # fixed size tensor , scalar -> fixed size tensor
        """
#trait_add = {{
 indexing_maps = [
   affine_map<(i, j) -> (i, j)>,
   affine_map<(i, j) -> (i, j)>
 ],
 iterator_types = [\"parallel\", \"parallel\"],
 doc = \" !!! DUMMY DOCUMENTATION !!! \"
}}
func @{func_name}(%arg_tensor: tensor<2x4x{mlir_type}>, %arg_scalar: {mlir_type}) -> tensor<2x4x{mlir_type}> {{
 %answer = linalg.generic #trait_add
    ins(%arg_tensor: tensor<2x4x{mlir_type}>)
   outs(%arg_tensor: tensor<2x4x{mlir_type}>) {{
     ^bb(%a: {mlir_type}, %s: {mlir_type}):
       %sum = {linalg_add} %a, %arg_scalar : {mlir_type}
       linalg.yield %sum : {mlir_type}
 }} -> tensor<2x4x{mlir_type}>
 return %answer : tensor<2x4x{mlir_type}>
}}
""",
        [np.arange(2 * 4).reshape((2, 4)), 2],
        np.array([[2, 3, 4, 5], [6, 7, 8, 9]]),
    ),
    (  # scalar , fixed size tensor -> fixed size tensor
        """
#trait_add = {{
 indexing_maps = [
   affine_map<(i, j) -> (i, j)>,
   affine_map<(i, j) -> (i, j)>
 ],
 iterator_types = [\"parallel\", \"parallel\"],
 doc = \" !!! DUMMY DOCUMENTATION !!! \"
}}
func @{func_name}(%arg_scalar: {mlir_type}, %arg_tensor: tensor<2x4x{mlir_type}>) -> tensor<2x4x{mlir_type}> {{
 %answer = linalg.generic #trait_add
    ins(%arg_tensor: tensor<2x4x{mlir_type}>)
   outs(%arg_tensor: tensor<2x4x{mlir_type}>) {{
     ^bb(%a: {mlir_type}, %s: {mlir_type}):
       %sum = {linalg_add} %a, %arg_scalar : {mlir_type}
       linalg.yield %sum : {mlir_type}
 }} -> tensor<2x4x{mlir_type}>
 return %answer : tensor<2x4x{mlir_type}>
}}
""",
        [2, np.arange(2 * 4).reshape((2, 4))],
        np.array([[2, 3, 4, 5], [6, 7, 8, 9]]),
    ),
    (  # fixed size tensor , scalar , fixed size tensor -> fixed size tensor
        """
#trait_add = {{
 indexing_maps = [
   affine_map<(i, j) -> (i, j)>,
   affine_map<(i, j) -> (i, j)>,
   affine_map<(i, j) -> (i, j)>
 ],
 iterator_types = [\"parallel\", \"parallel\"],
 doc = \" !!! DUMMY DOCUMENTATION !!! \"
}}
func @{func_name}(%tensor_a: tensor<2x4x{mlir_type}>, %arg_scalar: {mlir_type}, %tensor_b: tensor<2x4x{mlir_type}>) -> tensor<2x4x{mlir_type}> {{
 %answer = linalg.generic #trait_add
    ins(%tensor_a, %tensor_b: tensor<2x4x{mlir_type}>, tensor<2x4x{mlir_type}>)
   outs(%tensor_a: tensor<2x4x{mlir_type}>) {{
     ^bb(%a: {mlir_type}, %b: {mlir_type}, %s: {mlir_type}):
       %ab = {linalg_add} %a, %b : {mlir_type}
       %sum = {linalg_add} %ab, %arg_scalar : {mlir_type}
       linalg.yield %sum : {mlir_type}
 }} -> tensor<2x4x{mlir_type}>
 return %answer : tensor<2x4x{mlir_type}>
}}
""",
        [np.arange(2 * 4).reshape((2, 4)), 2, np.full([2, 4], 10)],
        np.array([[12, 13, 14, 15], [16, 17, 18, 19]]),
    ),
]


def test_jit_engine_tensor_result():

    engine = mlir_graphblas.MlirJitEngine()
    for test_case_index, test_case in enumerate(TENSOR_RESULT_TEST_CASES):
        for mlir_type, np_type in MLIR_TYPE_TO_NP_TYPE.items():
            func_name = f"func_{test_case_index}_{mlir_type}"

            if issubclass(np_type, np.integer):
                linalg_add = "addi"
            elif issubclass(np_type, np.floating):
                linalg_add = "addf"
            else:
                raise ValueError(f"No MLIR type for {np_type}.")

            mlir_template, args, expected_result = test_case
            mlir_text = mlir_template.format(
                func_name=func_name, mlir_type=mlir_type, linalg_add=linalg_add
            )
            args = [
                arg.astype(np_type) if isinstance(arg, np.ndarray) else arg
                for arg in args
            ]
            expected_result = (
                expected_result.astype(np_type)
                if isinstance(expected_result, np.ndarray)
                else expected_result
            )

            engine.add(mlir_text, STANDARD_PASSES)

            compiled_func = engine[func_name]
            result = compiled_func(*args)
            assert np.all(
                result == expected_result
            ), f"""
Input MLIR: 
{mlir_text}

Inputs: {args}
Result: {result}
Expected Result: {expected_result}
"""
