from .jit_engine_test_utils import MLIR_TYPE_TO_NP_TYPE, STANDARD_PASSES

import mlir_graphblas
import mlir_graphblas.sparse_utils
import pytest
import numpy as np

SIMPLE_TEST_CASES = [  # elements are ( mlir_template, args, expected_result )
    pytest.param(  # arbitrary size tensor , arbitrary size tensor -> arbitrary size tensor
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
func @{func_name}(%arga: tensor<?x?x{mlir_type}>, %argb: tensor<?x?x{mlir_type}>) -> tensor<?x?x{mlir_type}> {{
 %answer = linalg.generic #trait_add
    ins(%arga, %argb: tensor<?x?x{mlir_type}>, tensor<?x?x{mlir_type}>)
   outs(%arga: tensor<?x?x{mlir_type}>) {{
     ^bb(%a: {mlir_type}, %b: {mlir_type}, %s: {mlir_type}):
       %sum = {linalg_add} %a, %b : {mlir_type}
       linalg.yield %sum : {mlir_type}
 }} -> tensor<?x?x{mlir_type}>
 return %answer : tensor<?x?x{mlir_type}>
}}
""",
        [np.arange(2 * 4).reshape((2, 4)), np.full([2, 4], 10)],
        np.array([[10, 11, 12, 13], [14, 15, 16, 17]]),
        id="arbitrary_arbitrary_to_arbitrary",
    ),
    pytest.param(  # fixed size tensor , fixed size tensor -> fixed size tensor
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
        [np.arange(2 * 4).reshape((2, 4)), np.full([2, 4], 10)],
        np.array([[10, 11, 12, 13], [14, 15, 16, 17]]),
        id="fixed_fixed_to_fixed",
    ),
    pytest.param(  # arbitrary size tensor , fixed size tensor -> arbitrary size tensor
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
func @{func_name}(%arga: tensor<?x?x{mlir_type}>, %argb: tensor<2x4x{mlir_type}>) -> tensor<?x?x{mlir_type}> {{
 %answer = linalg.generic #trait_add
    ins(%arga, %argb: tensor<?x?x{mlir_type}>, tensor<2x4x{mlir_type}>)
   outs(%arga: tensor<?x?x{mlir_type}>) {{
     ^bb(%a: {mlir_type}, %b: {mlir_type}, %s: {mlir_type}):
       %sum = {linalg_add} %a, %b : {mlir_type}
       linalg.yield %sum : {mlir_type}
 }} -> tensor<?x?x{mlir_type}>
 return %answer : tensor<?x?x{mlir_type}>
}}
""",
        [np.arange(2 * 4).reshape((2, 4)), np.full([2, 4], 10)],
        np.array([[10, 11, 12, 13], [14, 15, 16, 17]]),
        id="arbitrary_fixed_to_arbitrary",
    ),
    pytest.param(  # fixed size tensor , scalar -> fixed size tensor
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
        id="fixed_scalar_to_fixed",
    ),
    pytest.param(  # fixed size tensor , scalar , arbitrary size tensor -> fixed size tensor
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
func @{func_name}(%tensor_a: tensor<2x4x{mlir_type}>, %arg_scalar: {mlir_type}, %tensor_b: tensor<?x?x{mlir_type}>) -> tensor<2x4x{mlir_type}> {{
 %answer = linalg.generic #trait_add
    ins(%tensor_a, %tensor_b: tensor<2x4x{mlir_type}>, tensor<?x?x{mlir_type}>)
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
        id="fixed_scalar_arbitrary_to_arbitrary",
    ),
    pytest.param(  # arbitrary size 1D tensor -> scalar
        """
func @{func_name}(%arg0: tensor<?x{mlir_type}>) -> {mlir_type} {{
  %c3 = constant 3 : index
  %ans = tensor.extract %arg0[%c3] : tensor<?x{mlir_type}>
  return %ans : {mlir_type}
}}
""",
        [np.arange(8)],
        3,
        id="arbitrary_to_scalar",
    ),
    pytest.param(  # arbitrary size 1D tensor, scalar -> scalar
        """
func @{func_name}(%tensor_arg: tensor<?x{mlir_type}>, %scalar_arg: {mlir_type}) -> {mlir_type} {{
  %c3 = constant 3 : index
  %element_3 = tensor.extract %tensor_arg[%c3] : tensor<?x{mlir_type}>
  %ans = {linalg_add} %element_3, %scalar_arg : {mlir_type}
  return %ans : {mlir_type}
}}
""",
        [np.arange(8), 10],
        13,
        id="arbitrary_scalar_to_scalar",
    ),
    pytest.param(  # arbitrary size 1D tensor, arbitrary size 1D tensor -> scalar
        """
func @{func_name}(%arg0: tensor<?x{mlir_type}>, %arg1: tensor<?x{mlir_type}>) -> {mlir_type} {{
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %arg0_3 = tensor.extract %arg0[%c3] : tensor<?x{mlir_type}>
  %arg1_4 = tensor.extract %arg1[%c4] : tensor<?x{mlir_type}>
  %ans = {linalg_add} %arg0_3, %arg1_4 : {mlir_type}
  return %ans : {mlir_type}
}}
""",
        [np.arange(8), np.arange(5)],
        7,
        id="arbitrary_arbitrary_to_scalar",
    ),
    pytest.param(  # scalar, arbitrary size 1D tensor, arbitrary size 1D tensor -> scalar
        """
func @{func_name}(%scalar: {mlir_type}, %arg0: tensor<?x{mlir_type}>, %arg1: tensor<?x{mlir_type}>) -> {mlir_type} {{
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %arg0_3 = tensor.extract %arg0[%c3] : tensor<?x{mlir_type}>
  %arg1_4 = tensor.extract %arg1[%c4] : tensor<?x{mlir_type}>
  %tensor_element_result = {linalg_add} %arg0_3, %arg1_4 : {mlir_type}
  %ans = {linalg_add} %tensor_element_result, %scalar : {mlir_type}
  return %ans : {mlir_type}
}}
""",
        [2, np.arange(8), np.arange(5)],
        9,
        id="scalar_arbitrary_arbitrary_to_scalar",
    ),
    pytest.param(  # simple and nested type aliases
        """
!mlir_type_alias = type {mlir_type}
!mlir_tensor_type_alias = type tensor<?x!mlir_type_alias>

func @{func_name}(%scalar: {mlir_type}, %arg0: tensor<?x!mlir_type_alias>, %arg1: !mlir_tensor_type_alias) -> {mlir_type} {{
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  %arg0_3 = tensor.extract %arg0[%c3] : tensor<?x{mlir_type}>
  %arg1_4 = tensor.extract %arg1[%c4] : tensor<?x{mlir_type}>
  %tensor_element_result = {linalg_add} %arg0_3, %arg1_4 : {mlir_type}
  %ans = {linalg_add} %tensor_element_result, %scalar : {mlir_type}
  return %ans : !mlir_type_alias
}}
""",
        [2, np.arange(8), np.arange(5)],
        9,
        id="simple_and_nested_type_aliases",
    ),
    pytest.param(  # partially fixed size tensor -> scalar
        """
func @{func_name}(%arg_tensor: tensor<?x4x{mlir_type}>) -> {mlir_type} {{
  %c0 = constant 0 : index
  %ans = tensor.extract %arg_tensor[%c0, %c0] : tensor<?x4x{mlir_type}>
  return %ans : {mlir_type}
}}
""",
        [np.full([7, 4], 100)],
        100,
        id="partial_to_scalar",
    ),
]


TEST_CASE_INDEX = 0


@pytest.mark.parametrize("mlir_template,args,expected_result", SIMPLE_TEST_CASES)
@pytest.mark.parametrize("mlir_type", MLIR_TYPE_TO_NP_TYPE.keys())
def test_jit_engine_tensor_result(
    engine, mlir_template, args, expected_result, mlir_type
):
    global TEST_CASE_INDEX

    np_type = MLIR_TYPE_TO_NP_TYPE[mlir_type]
    func_name = f"func_{TEST_CASE_INDEX}_{mlir_type}"
    TEST_CASE_INDEX += 1

    if issubclass(np_type, np.integer):
        linalg_add = "addi"
    elif issubclass(np_type, np.floating):
        linalg_add = "addf"
    else:
        raise ValueError(f"No MLIR type for {np_type}.")

    mlir_text = mlir_template.format(
        func_name=func_name, mlir_type=mlir_type, linalg_add=linalg_add
    )
    args = [arg.astype(np_type) if isinstance(arg, np.ndarray) else arg for arg in args]
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


@pytest.mark.parametrize("mlir_type", ["f32", "f64"])
def test_jit_engine_sparse_tensor(engine, mlir_type):
    mlir_template = r"""

#trait_sum_reduction = {{
  indexing_maps = [
    affine_map<(i,j,k) -> (i,j,k)>,  // A
    affine_map<(i,j,k) -> ()>        // x (scalar out)
  ],
  sparse = [
    [ "S", "S", "S" ],  // A
    [ ]                 // x
  ],
  iterator_types = ["reduction", "reduction", "reduction"],
  doc = "x += SUM_ijk A(i,j,k)"
}}

!SparseTensor = type !llvm.ptr<i8>

func @{func_name}(%argA: !SparseTensor) -> {mlir_type} {{
  %out_tensor = constant dense<0.0> : tensor<{mlir_type}>
  %arga = linalg.sparse_tensor %argA : !SparseTensor to tensor<10x20x30x{mlir_type}>
  %reduction = linalg.generic #trait_sum_reduction
     ins(%arga: tensor<10x20x30x{mlir_type}>)
    outs(%out_tensor: tensor<{mlir_type}>) {{
      ^bb(%a: {mlir_type}, %x: {mlir_type}):
        %0 = addf %x, %a : {mlir_type}
        linalg.yield %0 : {mlir_type}
  }} -> tensor<{mlir_type}>
  %answer = tensor.extract %reduction[] : tensor<{mlir_type}>
  return %answer : {mlir_type}
}}

"""

    np_type = MLIR_TYPE_TO_NP_TYPE[mlir_type]

    func_name = f"func_{mlir_type}"

    mlir_text = mlir_template.format(func_name=func_name, mlir_type=mlir_type)

    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint64)
    values = np.array([1.2, 3.4], dtype=np_type)
    sizes = np.array([10, 20, 30], dtype=np.uint64)
    sparsity = np.array([True, True, True], dtype=np.bool8)

    a = mlir_graphblas.sparse_utils.MLIRSparseTensor(indices, values, sizes, sparsity)

    assert engine.add(mlir_text, STANDARD_PASSES) == [func_name]

    result = engine[func_name](a)
    expected_result = 4.6
    assert (
        abs(result - expected_result) < 1e-6
    ), f"""
Input MLIR: 
{mlir_text}

Inputs: {args}
Result: {result}
Expected Result: {expected_result}
"""


@pytest.fixture(scope="module")
def engine():
    return mlir_graphblas.MlirJitEngine()
