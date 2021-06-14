import datetime
import mlir
import itertools
import pytest
import numpy as np

from mlir_graphblas import MlirJitEngine
from mlir_graphblas.engine import parse_mlir_string
from mlir_graphblas.sparse_utils import MLIRSparseTensor
from mlir_graphblas.mlir_builder import MLIRVar, MLIRFunctionBuilder
from mlir_graphblas.functions import ConvertLayout
from mlir_graphblas.algorithms import (
    triangle_count_combined,
    dense_neural_network_combined,
)

from .jit_engine_test_utils import sparsify_array, GRAPHBLAS_PASSES

from typing import List, Callable

# TODO a lot of these tests take sums or reductions over an scf.for loop by storing into a memref
# It's better practice to use as demonstrated
# at https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-mlirscfforop


@pytest.fixture(scope="module")
def engine():
    jit_engine = MlirJitEngine()

    jit_engine.add(
        """
#trait_densify_csr = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,
    affine_map<(i,j) -> (i,j)>
  ],
  iterator_types = ["parallel", "parallel"]
}

#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func @csr_densify8x8(%argA: tensor<8x8xf64, #CSR64>) -> tensor<8x8xf64> {
  %output_storage = constant dense<0.0> : tensor<8x8xf64>
  %0 = linalg.generic #trait_densify_csr
    ins(%argA: tensor<8x8xf64, #CSR64>)
    outs(%output_storage: tensor<8x8xf64>) {
      ^bb(%A: f64, %x: f64):
        linalg.yield %A : f64
    } -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

#trait_densify_csc = {
  indexing_maps = [
    affine_map<(i,j) -> (j,i)>,
    affine_map<(i,j) -> (i,j)>
  ],
  iterator_types = ["parallel", "parallel"]
}

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func @csc_densify8x8(%argA: tensor<8x8xf64, #CSC64>) -> tensor<8x8xf64> {
  %output_storage = constant dense<0.0> : tensor<8x8xf64>
  %0 = linalg.generic #trait_densify_csc
    ins(%argA: tensor<8x8xf64, #CSC64>)
    outs(%output_storage: tensor<8x8xf64>) {
      ^bb(%A: f64, %x: f64):
        linalg.yield %A : f64
    } -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}
""",
        GRAPHBLAS_PASSES,
    )
    return jit_engine


def test_ir_builder_convert_layout_wrapper(engine: MlirJitEngine):
    # Build Function
    convert_layout_function = ConvertLayout()

    input_var = MLIRVar("input_tensor", "tensor<?x?xf64, #CSR64>")

    ir_builder = MLIRFunctionBuilder(
        "convert_layout_wrapper",
        input_vars=[input_var],
        return_types=("tensor<?x?xf64, #CSC64>",),
    )
    convert_layout_result = ir_builder.call(convert_layout_function, input_var)
    ir_builder.return_vars(convert_layout_result)

    assert ir_builder.get_mlir()

    # Test Compiled Function
    convert_layout_wrapper_callable = ir_builder.compile(
        engine=engine, passes=GRAPHBLAS_PASSES
    )

    indices = np.array(
        [
            [1, 2],
            [4, 3],
        ],
        dtype=np.uint64,
    )
    values = np.array([1.2, 4.3], dtype=np.float64)
    sizes = np.array([8, 8], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)

    input_tensor = MLIRSparseTensor(indices, values, sizes, sparsity)

    dense_input_tensor = np.zeros([8, 8], dtype=np.float64)
    dense_input_tensor[1, 2] = 1.2
    dense_input_tensor[4, 3] = 4.3
    assert np.isclose(dense_input_tensor, engine.csr_densify8x8(input_tensor)).all()

    output_tensor = convert_layout_wrapper_callable(input_tensor)

    assert np.isclose(dense_input_tensor, engine.csc_densify8x8(output_tensor)).all()

    return


def test_ir_builder_triple_convert_layout(engine: MlirJitEngine):
    # Build Function

    input_var = MLIRVar("input_tensor", "tensor<?x?xf64, #CSR64>")

    ir_builder = MLIRFunctionBuilder(
        "triple_convert_layout",
        input_vars=(input_var,),
        return_types=["tensor<?x?xf64, #CSC64>"],
    )
    # Use different instances of Tranpose to ideally get exactly one convert_layout helper in the final MLIR text
    inter1 = ir_builder.call(ConvertLayout("csc"), input_var)
    inter2 = ir_builder.call(ConvertLayout("csr"), inter1)
    return_var = ir_builder.call(ConvertLayout("csc"), inter2)
    ir_builder.return_vars(return_var)

    mlir_text = ir_builder.get_mlir(make_private=False)
    ast = parse_mlir_string(mlir_text)
    # verify there are exactly two functions
    functions = [node for node in ast.body if isinstance(node, mlir.astnodes.Function)]
    triple_convert_func = functions.pop(-1)
    assert triple_convert_func.visibility == "public"
    assert all(
        func.name.value.startswith("convert_layout_cs") and func.visibility == "private"
        for func in functions
        if func.name.value == "triple_convert_layout"
    )

    # verify in_place_triple_convert_layout has three convert_layout calls
    region = triple_convert_func.body
    (block,) = region.body
    call_op_1, call_op_2, call_op_3, return_op = block.body
    assert call_op_1.op.func.value == "convert_layout_to_csc"
    assert call_op_2.op.func.value == "convert_layout_to_csr"
    assert call_op_3.op.func.value == "convert_layout_to_csc"
    (return_op_type,) = [
        t for t in mlir.dialects.standard.ops if t.__name__ == "ReturnOperation"
    ]
    assert isinstance(return_op.op, return_op_type)

    # Test Compiled Function
    triple_convert_layout_callable = ir_builder.compile(
        engine=engine, passes=GRAPHBLAS_PASSES
    )

    indices = np.array(
        [
            [1, 2],
            [4, 3],
        ],
        dtype=np.uint64,
    )
    values = np.array([1.2, 4.3], dtype=np.float64)
    sizes = np.array([8, 8], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)

    input_tensor = MLIRSparseTensor(indices, values, sizes, sparsity)

    dense_input_tensor = np.zeros([8, 8], dtype=np.float64)
    dense_input_tensor[1, 2] = 1.2
    dense_input_tensor[4, 3] = 4.3
    assert np.isclose(dense_input_tensor, engine.csr_densify8x8(input_tensor)).all()

    output_tensor = triple_convert_layout_callable(input_tensor)

    assert np.isclose(dense_input_tensor, engine.csc_densify8x8(output_tensor)).all()

    return


def test_ir_builder_triangle_count(engine: MlirJitEngine):
    # 0 - 1    5 - 6
    # | X |    | /
    # 3 - 4 -- 2 - 7
    # fmt: off
    indices = np.array(
        [
            [0, 1],
            [0, 3],
            [0, 4],
            [1, 0],
            [1, 3],
            [1, 4],
            [2, 4],
            [2, 5],
            [2, 6],
            [2, 7],
            [3, 0],
            [3, 1],
            [3, 4],
            [4, 0],
            [4, 1],
            [4, 2],
            [4, 3],
            [5, 2],
            [5, 6],
            [6, 2],
            [6, 5],
            [7, 2],
        ],
        dtype=np.uint64,
    )
    values = np.array([
        100, 200, 300, 100, 400, 500, 99, 50, 55, 75, 200,
        400, 600, 300, 500, 99, 600, 50, 60, 55, 60, 75],
        dtype=np.float64,
    )
    # fmt: on
    sizes = np.array([8, 8], dtype=np.uint64)
    sparsity = np.array([False, True], dtype=np.bool8)
    input_tensor = MLIRSparseTensor(indices, values, sizes, sparsity)

    assert 5 == triangle_count_combined(input_tensor)

    return


def test_ir_builder_for_loop_simple(engine: MlirJitEngine):
    # Build Function

    input_var = MLIRVar("input_value", "f64")

    ir_builder = MLIRFunctionBuilder(
        "times_three", input_vars=[input_var], return_types=["f64"]
    )
    zero_f64 = ir_builder.constant(0.0, "f64")
    ir_builder.add_statement(
        f"""
// pymlir-skip: begin
%sum_memref = memref.alloc() : memref<f64>
memref.store {zero_f64.access_string()}, %sum_memref[] : memref<f64>
"""
    )
    with ir_builder.for_loop(0, 3) as for_vars:
        ir_builder.add_statement(
            f"""
%current_sum = memref.load %sum_memref[] : memref<f64>
%updated_sum = addf {input_var.access_string()}, %current_sum : f64
memref.store %updated_sum, %sum_memref[] : memref<f64>
"""
        )
    assert for_vars.returned_variable is None
    result_var = MLIRVar("sum", "f64")
    ir_builder.add_statement(
        f"""
{result_var.assign_string()} = memref.load %sum_memref[] : memref<f64>
// pymlir-skip: end
"""
    )
    ir_builder.return_vars(result_var)

    assert ir_builder.get_mlir()

    # Test Compiled Function
    times_three = ir_builder.compile(engine=engine)
    assert np.isclose(times_three(1.3), 3.9)

    return


def test_ir_builder_for_loop_float_iter(engine: MlirJitEngine):
    # Build Function

    lower_i = 0
    upper_i = 4
    delta_i = 1
    lower_float = 0.0
    delta_float = 7.8

    input_var = MLIRVar("input_value", "f64")

    ir_builder = MLIRFunctionBuilder(
        "plus_6x7_8", input_vars=[input_var], return_types=["f64"]
    )
    ir_builder.add_statement(
        f"""
// pymlir-skip: begin
%sum_memref = memref.alloc() : memref<f64>
memref.store {input_var.access_string()}, %sum_memref[] : memref<f64>
"""
    )

    float_lower_var = ir_builder.constant(lower_float, "f64")
    float_iter_var = MLIRVar("float_iter", "f64")
    float_delta_var = ir_builder.constant(delta_float, "f64")
    with ir_builder.for_loop(
        lower_i, upper_i, delta_i, iter_vars=[(float_iter_var, float_lower_var)]
    ) as for_vars:
        assert [float_iter_var] == for_vars.iter_vars
        incremented_float_var = MLIRVar("incremented_float", "f64")
        ir_builder.add_statement(
            f"""
%current_sum = memref.load %sum_memref[] : memref<f64>
%updated_sum = addf {float_iter_var.access_string()}, %current_sum : f64
memref.store %updated_sum, %sum_memref[] : memref<f64>
{incremented_float_var.assign_string()} = addf {float_iter_var.access_string()}, {float_delta_var.access_string()} : f64
"""
        )
        for_vars.yield_vars(incremented_float_var)
    result_var = MLIRVar("sum", "f64")
    ir_builder.add_statement(
        f"""
{result_var.assign_string()} = memref.load %sum_memref[] : memref<f64>
// pymlir-skip: end
"""
    )
    ir_builder.return_vars(result_var)

    assert ir_builder.get_mlir()

    # Test Compiled Function
    func = ir_builder.compile(engine=engine)
    expected_sum = 1.3 + sum(range(lower_i, upper_i, delta_i)) * delta_float
    assert np.isclose(func(1.3), expected_sum)

    return


def test_ir_builder_for_loop_user_specified_vars(engine: MlirJitEngine):
    # Build Function

    input_var = MLIRVar("input_value", "i64")

    lower_index = 3
    upper_index = 9
    delta_index = 2
    lower_i64 = 5
    delta_i64 = 7

    # this expected_sum calculation is expected to be isomorphic to the generated MLIR
    expected_sum = 7
    index_iterator = range(lower_index, upper_index, delta_index)
    i64_iterator = itertools.count(lower_i64, delta_i64)
    for iter_index, iter_i64 in zip(index_iterator, i64_iterator):
        expected_sum += lower_index * upper_index * delta_index
        expected_sum += lower_i64 * delta_i64
        expected_sum += iter_index * iter_i64

    # Build IR
    ir_builder = MLIRFunctionBuilder(
        "add_user_specified_vars", input_vars=[input_var], return_types=["i64"]
    )
    ir_builder.add_statement(
        f"""
// pymlir-skip: begin
%sum_memref = memref.alloc() : memref<i64>
memref.store {input_var.access_string()}, %sum_memref[] : memref<i64>
"""
    )
    lower_index_var = ir_builder.constant(lower_index, "index")
    upper_index_var = ir_builder.constant(upper_index, "index")
    delta_index_var = ir_builder.constant(delta_index, "index")
    lower_i64_var = ir_builder.constant(lower_i64, "i64")
    delta_i64_var = ir_builder.constant(delta_i64, "i64")
    iter_i64_var = MLIRVar("iter_i64", "i64")

    with ir_builder.for_loop(
        lower_index_var,
        upper_index_var,
        delta_index_var,
        iter_vars=[(iter_i64_var, lower_i64_var)],
    ) as for_vars:
        assert lower_index_var == for_vars.lower_var_index
        assert upper_index_var == for_vars.upper_var_index
        assert delta_index_var == for_vars.step_var_index
        assert [iter_i64_var] == for_vars.iter_vars
        incremented_iter_i64_var = MLIRVar("incremented_iter_i64", "i64")
        ir_builder.add_statement(
            f"""
%current_sum = memref.load %sum_memref[] : memref<i64>
%prod_of_index_vars_0 = muli {for_vars.lower_var_index.access_string()}, {for_vars.upper_var_index.access_string()} : index
%prod_of_index_vars_1 = muli %prod_of_index_vars_0, {for_vars.step_var_index.access_string()} : index
%prod_of_index_vars = std.index_cast %prod_of_index_vars_1 : index to i64
%prod_of_i64_vars = muli {lower_i64_var.access_string()}, {delta_i64_var.access_string()} : i64
%iter_index_i64 = std.index_cast {for_vars.iter_var_index.access_string()} : index to i64
%prod_of_iter_vars = muli %iter_index_i64, {iter_i64_var.access_string()} : i64
%updated_sum_0 = addi %current_sum, %prod_of_index_vars : i64
%updated_sum_1 = addi %updated_sum_0, %prod_of_i64_vars : i64
%updated_sum = addi %updated_sum_1, %prod_of_iter_vars : i64
memref.store %updated_sum, %sum_memref[] : memref<i64>
{incremented_iter_i64_var.assign_string()} = addi {iter_i64_var.access_string()}, {delta_i64_var.access_string()} : i64
"""
        )
        for_vars.yield_vars(incremented_iter_i64_var)
    result_var = MLIRVar("sum", "i64")
    ir_builder.add_statement(
        f"""
{result_var.assign_string()} = memref.load %sum_memref[] : memref<i64>
// pymlir-skip: end
"""
    )
    ir_builder.return_vars(result_var)

    assert (
        ir_builder.get_mlir()
    )  # this generated MLIR is easier to read than the above IR builder calls.

    # Test Compiled Function
    func = ir_builder.compile(engine=engine)

    calculated_sum = func(7)
    assert np.isclose(calculated_sum, expected_sum)

    return


DNN_CASES = [
    pytest.param(
        lambda *args: np.arange(*args) / 100.0,
        10,
        1_000_000.0,
        id="arange_10_layer",
    ),
    pytest.param(
        lambda *args: np.random.rand(*args) / 100.0,
        10,
        32.0,
        id="random_10_layer",
    ),
]


@pytest.mark.parametrize(
    "array_initializer, max_num_layers, clamp_threshold", DNN_CASES
)
def test_ir_builder_dnn(
    engine: MlirJitEngine,
    array_initializer: Callable,
    max_num_layers: int,
    clamp_threshold: float,
):
    sparsify_matrix = lambda matrix: sparsify_array(matrix, [False, True])
    np.random.seed(hash(datetime.date.today()) % 2 ** 32)

    for num_layers in range(1, max_num_layers + 1):

        dense_weight_matrices = [array_initializer(64) for _ in range(num_layers)]
        dense_weight_matrices = [m.reshape(8, 8) for m in dense_weight_matrices]
        dense_weight_matrices = [m.astype(np.float64) for m in dense_weight_matrices]

        dense_bias_vectors = [array_initializer(8) for _ in range(num_layers)]
        dense_bias_vectors = [vec.astype(np.float64) for vec in dense_bias_vectors]
        dense_bias_matrices = [np.diag(vec) for vec in dense_bias_vectors]
        dense_bias_matrices = [np.nan_to_num(m) for m in dense_bias_matrices]

        dense_input_tensor = array_initializer(64).reshape(8, 8).astype(np.float64)

        numpy_dense_result = dense_input_tensor
        for dense_weight_matrix, dense_bias_vector in zip(
            dense_weight_matrices, dense_bias_vectors
        ):
            numpy_dense_result = numpy_dense_result @ dense_weight_matrix
            numpy_dense_result = numpy_dense_result + dense_bias_vector
            numpy_dense_result = numpy_dense_result * (dense_bias_vector != 0).astype(
                int
            )  # TODO is this correct?
            numpy_dense_result = numpy_dense_result * (numpy_dense_result > 0).astype(
                int
            )
            numpy_dense_result[numpy_dense_result > clamp_threshold] = clamp_threshold

        sparse_weight_matrices = [
            sparsify_matrix(matrix) for matrix in dense_weight_matrices
        ]
        sparse_bias_matrices = [
            sparsify_matrix(matrix) for matrix in dense_bias_matrices
        ]
        sparse_input_tensor = sparsify_matrix(dense_input_tensor)
        sparse_result = dense_neural_network_combined(
            sparse_weight_matrices,
            sparse_bias_matrices,
            sparse_input_tensor,
            clamp_threshold,
        )
        dense_result = engine.csr_densify8x8(sparse_result)

        with np.printoptions(suppress=True):
            assert np.isclose(
                dense_result, numpy_dense_result
            ).all(), f"""
num_layers
{num_layers}

dense_input_tensor
{dense_input_tensor}

dense_result
{dense_result}

numpy_dense_result
{numpy_dense_result}

np.isclose(dense_result, numpy_dense_result)
{np.isclose(dense_result, numpy_dense_result)}
"""

    return
