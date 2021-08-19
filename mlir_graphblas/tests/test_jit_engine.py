from .jit_engine_test_utils import MLIR_TYPE_TO_NP_TYPE, STANDARD_PASSES

import itertools
import mlir_graphblas
from mlir_graphblas.sparse_utils import MLIRSparseTensor
import pytest
import numpy as np

SIMPLE_TEST_CASES = [  # elements are ( mlir_template, args, expected_result )
    # TODO: figure out why this is failing (failed to legalize operation 'memref.dim')
    #     pytest.param(  # arbitrary size tensor , arbitrary size tensor -> arbitrary size tensor
    #         """
    # #trait_add = {{
    #  indexing_maps = [
    #    affine_map<(i, j) -> (i, j)>,
    #    affine_map<(i, j) -> (i, j)>,
    #    affine_map<(i, j) -> (i, j)>
    #  ],
    #  iterator_types = [\"parallel\", \"parallel\"],
    #  doc = \" !!! DUMMY DOCUMENTATION !!! \"
    # }}
    # func @{func_name}(%arga: tensor<?x?x{mlir_type}>, %argb: tensor<?x?x{mlir_type}>) -> tensor<?x?x{mlir_type}> {{
    #  %answer = linalg.generic #trait_add
    #     ins(%arga, %argb: tensor<?x?x{mlir_type}>, tensor<?x?x{mlir_type}>)
    #    outs(%arga: tensor<?x?x{mlir_type}>) {{
    #      ^bb(%a: {mlir_type}, %b: {mlir_type}, %s: {mlir_type}):
    #        %sum = {std_add} %a, %b : {mlir_type}
    #        linalg.yield %sum : {mlir_type}
    #  }} -> tensor<?x?x{mlir_type}>
    #  return %answer : tensor<?x?x{mlir_type}>
    # }}
    # """,
    #         [np.arange(2 * 4).reshape((2, 4)), np.full([2, 4], 10)],
    #         np.array([[10, 11, 12, 13], [14, 15, 16, 17]]),
    #         id="arbitrary_arbitrary_to_arbitrary",
    #     ),
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
       %sum = {std_add} %a, %b : {mlir_type}
       linalg.yield %sum : {mlir_type}
 }} -> tensor<2x4x{mlir_type}>
 return %answer : tensor<2x4x{mlir_type}>
}}
""",
        [np.arange(2 * 4).reshape((2, 4)), np.full([2, 4], 10)],
        np.array([[10, 11, 12, 13], [14, 15, 16, 17]]),
        id="fixed_fixed_to_fixed",
    ),
    # TODO: figure out why this is failing (failed to legalize operation 'memref.dim')
    #     pytest.param(  # arbitrary size tensor , fixed size tensor -> arbitrary size tensor
    #         """
    # #trait_add = {{
    #  indexing_maps = [
    #    affine_map<(i, j) -> (i, j)>,
    #    affine_map<(i, j) -> (i, j)>,
    #    affine_map<(i, j) -> (i, j)>
    #  ],
    #  iterator_types = [\"parallel\", \"parallel\"],
    #  doc = \" !!! DUMMY DOCUMENTATION !!! \"
    # }}
    # func @{func_name}(%arga: tensor<?x?x{mlir_type}>, %argb: tensor<2x4x{mlir_type}>) -> tensor<?x?x{mlir_type}> {{
    #  %answer = linalg.generic #trait_add
    #     ins(%arga, %argb: tensor<?x?x{mlir_type}>, tensor<2x4x{mlir_type}>)
    #    outs(%arga: tensor<?x?x{mlir_type}>) {{
    #      ^bb(%a: {mlir_type}, %b: {mlir_type}, %s: {mlir_type}):
    #        %sum = {std_add} %a, %b : {mlir_type}
    #        linalg.yield %sum : {mlir_type}
    #  }} -> tensor<?x?x{mlir_type}>
    #  return %answer : tensor<?x?x{mlir_type}>
    # }}
    # """,
    #         [np.arange(2 * 4).reshape((2, 4)), np.full([2, 4], 10)],
    #         np.array([[10, 11, 12, 13], [14, 15, 16, 17]]),
    #         id="arbitrary_fixed_to_arbitrary",
    #     ),
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
       %sum = {std_add} %a, %arg_scalar : {mlir_type}
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
       %ab = {std_add} %a, %b : {mlir_type}
       %sum = {std_add} %ab, %arg_scalar : {mlir_type}
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
  %ans = {std_add} %element_3, %scalar_arg : {mlir_type}
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
  %ans = {std_add} %arg0_3, %arg1_4 : {mlir_type}
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
  %tensor_element_result = {std_add} %arg0_3, %arg1_4 : {mlir_type}
  %ans = {std_add} %tensor_element_result, %scalar : {mlir_type}
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
  %tensor_element_result = {std_add} %arg0_3, %arg1_4 : {mlir_type}
  %ans = {std_add} %tensor_element_result, %scalar : {mlir_type}
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

TEST_CASE_ID_GENERATOR = itertools.count()


@pytest.mark.parametrize("mlir_template,args,expected_result", SIMPLE_TEST_CASES)
@pytest.mark.parametrize("mlir_type", MLIR_TYPE_TO_NP_TYPE.keys())
def test_jit_engine_simple(engine, mlir_template, args, expected_result, mlir_type):
    np_type = MLIR_TYPE_TO_NP_TYPE[mlir_type]
    func_name = f"func_{next(TEST_CASE_ID_GENERATOR)}_{mlir_type}"

    std_operations_prefixes = ("add", "sub", "mul")
    if issubclass(np_type, np.integer):
        std_operations = {f"std_{op}": f"{op}i" for op in std_operations_prefixes}
    elif issubclass(np_type, np.floating):
        std_operations = {f"std_{op}": f"{op}f" for op in std_operations_prefixes}
    else:
        raise ValueError(f"No MLIR type for {np_type}.")

    mlir_text = mlir_template.format(
        func_name=func_name, mlir_type=mlir_type, **std_operations
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
  iterator_types = ["reduction", "reduction", "reduction"],
  doc = "x += SUM_ijk A(i,j,k)"
}}

#sparseTensor = #sparse_tensor.encoding<{{
  dimLevelType = [ "compressed", "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j,k) -> (i,j,k)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}}>

func @{func_name}(%argA: tensor<10x20x30x{mlir_type}, #sparseTensor>) -> {mlir_type} {{
  %out_tensor = constant dense<0.0> : tensor<{mlir_type}>
  %reduction = linalg.generic #trait_sum_reduction
     ins(%argA: tensor<10x20x30x{mlir_type}, #sparseTensor>)
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

    a = MLIRSparseTensor(indices, values, sizes, sparsity)

    assert engine.add(mlir_text, STANDARD_PASSES) == [func_name]

    result = engine[func_name](a)
    expected_result = 4.6
    assert (
        abs(result - expected_result) < 1e-6
    ), f"""
Input MLIR: 
{mlir_text}

Inputs: {values}
Result: {result}
Expected Result: {expected_result}
"""


@pytest.mark.parametrize("mlir_type", ["f32", "f64"])
def test_jit_engine_singleton_tuple_return_value(engine, mlir_type):
    mlir_text = f"""
func @func_singleton_tuple_return_value_{mlir_type}() -> ({mlir_type}) {{
    %result_0 = constant 12.34 : {mlir_type}
    return %result_0 : {mlir_type}
}}
"""

    assert engine.add(mlir_text, STANDARD_PASSES) == [
        f"func_singleton_tuple_return_value_{mlir_type}"
    ]
    result = engine[f"func_singleton_tuple_return_value_{mlir_type}"]()
    # Singleton tuple is removed during mlir-opt parsing normalization step
    expected_type = {
        "f32": np.float32,
        "f64": float,
    }[mlir_type]
    assert isinstance(result, expected_type)
    assert np.isclose(result, 12.34)

    return


def test_jit_engine_multiple_scalar_return_values(engine):
    mlir_text = """
func @func_multiple_scalar_return_values() -> (i8, i16, i32, i64, f32, f64) {
    %result_0 = constant 8 : i8
    %result_1 = constant 16 : i16
    %result_2 = constant 32 : i32
    %result_3 = constant 64 : i64
    %result_4 = constant 12.34 : f32
    %result_5 = constant 5.6e200 : f64
    return %result_0, %result_1, %result_2, %result_3, %result_4, %result_5 : i8, i16, i32, i64, f32, f64
}
"""
    assert engine.add(mlir_text, STANDARD_PASSES) == [
        "func_multiple_scalar_return_values"
    ]
    results = engine.func_multiple_scalar_return_values()
    assert results[0] == 8
    assert results[1] == 16
    assert results[2] == 32
    assert results[3] == 64
    assert np.isclose(results[4], 12.34)
    assert np.isclose(results[5], 5.6e200)

    return


def test_jit_engine_multiple_dense_tensor_return_values(engine):
    mlir_text = """
func @func_multiple_dense_tensor_return_values(%dummy_input_a: tensor<1xi8>, %dummy_input_b: i64) -> (f32, tensor<2x2xf32>, tensor<3x3xi8>) {
    %result_0 = constant 12.34 : f32
    %result_1 = constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
    %result_2 = constant dense<[[1, 2, 3], [4, 5, 6], [7, 8, 9]]> : tensor<3x3xi8>
    return %result_0, %result_1, %result_2 : f32, tensor<2x2xf32>, tensor<3x3xi8>
}
"""
    assert engine.add(mlir_text, STANDARD_PASSES) == [
        "func_multiple_dense_tensor_return_values"
    ]
    dummy_input_a = np.array([111], dtype=np.int8)
    dummy_input_b = 222
    results = engine.func_multiple_dense_tensor_return_values(
        dummy_input_a, dummy_input_b
    )
    assert isinstance(results, tuple)
    assert len(results) == 3
    assert np.isclose(results[0], 12.34)
    assert np.isclose(
        results[1], np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ).all()
    assert np.all(
        results[2] == np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
    )

    return


@pytest.mark.parametrize("mlir_type", MLIR_TYPE_TO_NP_TYPE.keys())
def test_jit_engine_sequence_of_scalars_input(engine, mlir_type):
    if mlir_type == "i8":
        return
    # TODO make this case part of SIMPLE_TEST_CASES when !llvm.ptr<i8>
    # no longer must denote a sparse tensor
    mlir_template, args, expected_result = (  # sequence of scalars -> scalar
        """
func @{func_name}(%sequence: !llvm.ptr<{mlir_type}>) -> {mlir_type} {{
  %czero = constant {zero_str} : {mlir_type}

  %sum_memref = memref.alloc() : memref<{mlir_type}>
  memref.store %czero, %sum_memref[] : memref<{mlir_type}>
  
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %sequence_length = constant 5 : index // hard-coded length

  %ci0 = constant 0 : i64
  %ci1 = constant 1 : i64

  scf.for %i = %c0 to %sequence_length step %c1 iter_args(%iter=%ci0) -> (i64) {{
  
    // llvm.getelementptr just does pointer arithmetic
    %element_ptr = llvm.getelementptr %sequence[%iter] : (!llvm.ptr<{mlir_type}>, i64) -> !llvm.ptr<{mlir_type}>
    
    // dereference %element_ptr to get an !llvm.ptr<i8>
    %element = llvm.load %element_ptr : !llvm.ptr<{mlir_type}>
    
    %current_sum = memref.load %sum_memref[] : memref<{mlir_type}>
    %updated_sum = {std_add} %current_sum, %element : {mlir_type}
    memref.store %updated_sum, %sum_memref[] : memref<{mlir_type}>
  
    %plus_one = addi %iter, %ci1 : i64
    scf.yield %plus_one : i64
  }}
  
  %sum = memref.load %sum_memref[] : memref<{mlir_type}>
  

  return %sum : {mlir_type}
}}
""",
        [(2, 4, 6, 8, 10)],
        30,
        # id="sequence_to_scalar",
    )

    np_type = MLIR_TYPE_TO_NP_TYPE[mlir_type]
    func_name = f"func_{next(TEST_CASE_ID_GENERATOR)}_{mlir_type}"

    std_operations_prefixes = ("add", "sub", "mul")
    if issubclass(np_type, np.integer):
        std_operations = {f"std_{op}": f"{op}i" for op in std_operations_prefixes}
    elif issubclass(np_type, np.floating):
        std_operations = {f"std_{op}": f"{op}f" for op in std_operations_prefixes}
    else:
        raise ValueError(f"No MLIR type for {np_type}.")

    zero_str = "0.0" if mlir_type[0] == "f" else "0"

    mlir_text = mlir_template.format(
        func_name=func_name, mlir_type=mlir_type, zero_str=zero_str, **std_operations
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
    return


def test_jit_engine_sequence_of_sparse_tensors_input(engine):
    mlir_text = """
#trait_sum_reduction = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,
    affine_map<(i,j) -> ()>
  ],
  iterator_types = ["reduction", "reduction"],
  doc = "Sparse Tensor Sum"
}

#sparseTensor = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func private @ptr8_to_matrix(!llvm.ptr<i8>) -> tensor<2x3xf64, #sparseTensor>

func @sparse_tensors_summation(%sequence: !llvm.ptr<!llvm.ptr<i8>>, %sequence_length: index) -> f64 {
  // Take an array of sparse 2x3 matrices


  %output_storage = constant dense<0.0> : tensor<f64>

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %ci0 = constant 0 : i64
  %ci1 = constant 1 : i64
  %cf0 = constant 0.0 : f64

  %sum_memref = memref.alloc() : memref<f64>
  memref.store %cf0, %sum_memref[] : memref<f64>

  scf.for %i = %c0 to %sequence_length step %c1 iter_args(%iter=%ci0) -> (i64) {

    // llvm.getelementptr just does pointer arithmetic
    %sparse_tensor_ptr_ptr = llvm.getelementptr %sequence[%iter] : (!llvm.ptr<!llvm.ptr<i8>>, i64) -> !llvm.ptr<!llvm.ptr<i8>>

    // dereference %sparse_tensor_ptr_ptr to get an !llvm.ptr<i8>
    %sparse_tensor_ptr = llvm.load %sparse_tensor_ptr_ptr : !llvm.ptr<!llvm.ptr<i8>>

    %sparse_tensor = call @ptr8_to_matrix(%sparse_tensor_ptr) : (!llvm.ptr<i8>) -> tensor<2x3xf64, #sparseTensor>

    %reduction = linalg.generic #trait_sum_reduction
        ins(%sparse_tensor: tensor<2x3xf64, #sparseTensor>)
        outs(%output_storage: tensor<f64>) {
          ^bb(%a: f64, %x: f64):
            %0 = addf %x, %a : f64
            linalg.yield %0 : f64
      } -> tensor<f64>
    %reduction_value = tensor.extract %reduction[] : tensor<f64>

    %current_sum = memref.load %sum_memref[] : memref<f64>
    %updated_sum = addf %reduction_value, %current_sum : f64
    memref.store %updated_sum, %sum_memref[] : memref<f64>

    %plus_one = addi %iter, %ci1 : i64
    scf.yield %plus_one : i64
  }

  %sum = memref.load %sum_memref[] : memref<f64>


  return %sum : f64
}
"""
    assert engine.add(mlir_text, STANDARD_PASSES) == ["sparse_tensors_summation"]

    num_sparse_tensors = 10

    # generate values
    sqrt_values = np.sqrt(np.arange(1, num_sparse_tensors + 1, 0.5))
    value_iter = iter(sqrt_values)
    next_values = lambda: np.array(
        [next(value_iter), next(value_iter)], dtype=np.float64
    )

    # generate coordinates
    coordinate_iter = itertools.cycle(range(2 * 3))
    next_indices = lambda: np.array(
        [next(coordinate_iter) for _ in range(4)], dtype=np.uint64
    ).reshape([2, 2])

    sparse_tensors = []
    for _ in range(num_sparse_tensors):
        indices = next_indices()
        values = next_values()
        sizes = np.array([2, 3], dtype=np.uint64)
        sparsity = np.array([True, True], dtype=np.bool8)
        sparse_tensor = MLIRSparseTensor(indices, values, sizes, sparsity)
        sparse_tensors.append(sparse_tensor)

    expected_sum = sqrt_values.sum()
    assert np.isclose(
        expected_sum,
        engine.sparse_tensors_summation(sparse_tensors, num_sparse_tensors),
    )

    return


def test_jit_engine_zero_values(engine):

    mlir_text = """
    module  {
      func private @sparseValuesF64(!llvm.ptr<i8>) -> memref<?xf64>
      func private @sparseIndices64(!llvm.ptr<i8>, index) -> memref<?xindex>
      func private @sparsePointers64(!llvm.ptr<i8>, index) -> memref<?xindex>
      func private @sparseDimSize(!llvm.ptr<i8>, index) -> index

      func @transpose(%output: !llvm.ptr<i8>, %input: !llvm.ptr<i8>) {
        %c0 = constant 0 : index
        %c1 = constant 1 : index


        %n_row = call @sparseDimSize(%input, %c0) : (!llvm.ptr<i8>, index) -> index
        %n_col = call @sparseDimSize(%input, %c1) : (!llvm.ptr<i8>, index) -> index
        %Ap = call @sparsePointers64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Aj = call @sparseIndices64(%input, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Ax = call @sparseValuesF64(%input) : (!llvm.ptr<i8>) -> memref<?xf64>
        %Bp = call @sparsePointers64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bi = call @sparseIndices64(%output, %c1) : (!llvm.ptr<i8>, index) -> memref<?xindex>
        %Bx = call @sparseValuesF64(%output) : (!llvm.ptr<i8>) -> memref<?xf64>

        %nnz = memref.load %Ap[%n_row] : memref<?xindex>

        // compute number of non-zero entries per column of A
        scf.for %arg2 = %c0 to %n_col step %c1 {
          memref.store %c0, %Bp[%arg2] : memref<?xindex>
        }
        scf.for %n = %c0 to %nnz step %c1 {
          %colA = memref.load %Aj[%n] : memref<?xindex>
          %colB = memref.load %Bp[%colA] : memref<?xindex>
          %colB1 = addi %colB, %c1 : index
          memref.store %colB1, %Bp[%colA] : memref<?xindex>
        }

        // cumsum the nnz per column to get Bp
        memref.store %c0, %Bp[%n_col] : memref<?xindex>
        scf.for %col = %c0 to %n_col step %c1 {
          %temp = memref.load %Bp[%col] : memref<?xindex>
          %cumsum = memref.load %Bp[%n_col] : memref<?xindex>
          memref.store %cumsum, %Bp[%col] : memref<?xindex>
          %cumsum2 = addi %cumsum, %temp : index
          memref.store %cumsum2, %Bp[%n_col] : memref<?xindex>
        }

        scf.for %row = %c0 to %n_row step %c1 {
          %j_start = memref.load %Ap[%row] : memref<?xindex>
          %row_plus1 = addi %row, %c1 : index
          %j_end = memref.load %Ap[%row_plus1] : memref<?xindex>
          scf.for %jj = %j_start to %j_end step %c1 {
            %col = memref.load %Aj[%jj] : memref<?xindex>
            %dest = memref.load %Bp[%col] : memref<?xindex>

            memref.store %row, %Bi[%dest] : memref<?xindex>
            %axjj = memref.load %Ax[%jj] : memref<?xf64>
            memref.store %axjj, %Bx[%dest] : memref<?xf64>

            // Bp[col]++
            %bp_inc = memref.load %Bp[%col] : memref<?xindex>
            %bp_inc1 = addi %bp_inc, %c1 : index
            memref.store %bp_inc1, %Bp[%col]: memref<?xindex>
          }
        }

        %last_last = memref.load %Bp[%n_col] : memref<?xindex>
        memref.store %c0, %Bp[%n_col] : memref<?xindex>
        scf.for %col = %c0 to %n_col step %c1 {
          %temp = memref.load %Bp[%col] : memref<?xindex>
          %last = memref.load %Bp[%n_col] : memref<?xindex>
          memref.store %last, %Bp[%col] : memref<?xindex>
          memref.store %temp, %Bp[%n_col] : memref<?xindex>
        }
        memref.store %last_last, %Bp[%n_col] : memref<?xindex>

        return
      }
    }
    """

    indices = np.array(
        [
            [0, 0],
            [1, 0],
        ],
        dtype=np.uint64,
    )
    values = np.array([8, 9], dtype=np.float64)
    sizes = np.array([2, 2], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)

    input_tensor = MLIRSparseTensor(indices, values, sizes, sparsity)
    output_tensor = MLIRSparseTensor(indices, values, sizes, sparsity)

    assert engine.add(mlir_text, STANDARD_PASSES) == ["transpose"]
    assert engine.transpose(output_tensor, input_tensor) is None
    assert np.isclose(engine.densify2x2(input_tensor), np.array([[8, 0], [9, 0]])).all()
    assert np.isclose(
        engine.densify2x2(output_tensor), np.array([[8, 9], [0, 0]])
    ).all()

    return


def test_jit_engine_skip(engine):
    # scf.if cannot be parsed by pymlir currently
    mlir_code = r"""
func private @relu_scalar(%arg0: f32) -> f32 {
  %cst = constant 0.000000e+00 : f32
  %0 = cmpf uge, %arg0, %cst : f32
  %1 = scf.if %0 -> (f32) {
    scf.yield %arg0 : f32
  } else {
    scf.yield %cst : f32
  }
  return %1 : f32
}

func @test_func(%arg0: f32) -> f32 {
  return %arg0 : f32
}
"""

    assert engine.add(mlir_code, STANDARD_PASSES) == ["test_func"]
    assert engine["test_func"](4) == 4


@pytest.fixture(scope="module")
def engine():
    engine = mlir_graphblas.MlirJitEngine()
    engine.add(
        """
#trait_densify = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,
    affine_map<(i,j) -> (i,j)>
  ],
  iterator_types = ["parallel", "parallel"]
}

#sparseTensor = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func @densify2x2(%argA: tensor<2x2xf64, #sparseTensor>) -> tensor<2x2xf64> {
  %output_storage = constant dense<0.0> : tensor<2x2xf64>
  %0 = linalg.generic #trait_densify
    ins(%argA: tensor<2x2xf64, #sparseTensor>)
    outs(%output_storage: tensor<2x2xf64>) {
      ^bb(%A: f64, %x: f64):
        linalg.yield %A : f64
    } -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
""",
        STANDARD_PASSES,
    )
    return engine
