from collections import defaultdict
import datetime
import itertools
import pytest
import numpy as np

from mlir_graphblas import MlirJitEngine
from mlir_graphblas.sparse_utils import MLIRSparseTensor
from mlir_graphblas.random_utils import ChooseUniformContext, ChooseWeightedContext
from mlir_graphblas.mlir_builder import MLIRFunctionBuilder
from mlir_graphblas.types import (
    AliasMap,
    SparseEncodingType,
    SparseTensorType,
    AffineMap,
)
from mlir_graphblas.algorithms import (
    triangle_count,
    dense_neural_network,
)

from mlir_graphblas.tools.utils import sparsify_array

from .jit_engine_test_utils import MLIR_TYPE_TO_NP_TYPE
from mlir_graphblas.mlir_builder import GRAPHBLAS_PASSES

from typing import Callable


@pytest.fixture(scope="module")
def engine():
    jit_engine = MlirJitEngine()

    jit_engine.add(
        """
#CSR64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (i,j)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSC64 = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

func @csr_to_csc(%matrix: tensor<?x?xf64, #CSR64>) -> tensor<?x?xf64, #CSC64> {
  %converted = graphblas.convert_layout %matrix : tensor<?x?xf64, #CSR64> to tensor<?x?xf64, #CSC64>
  return %converted : tensor<?x?xf64, #CSC64>
}

""",
        GRAPHBLAS_PASSES,
    )

    return jit_engine


@pytest.fixture(scope="module")
def aliases() -> AliasMap:
    csr64 = SparseEncodingType(["dense", "compressed"], [0, 1], 64, 64)
    csc64 = SparseEncodingType(["dense", "compressed"], [1, 0], 64, 64)
    cv64 = SparseEncodingType(["compressed"], None, 64, 64)
    aliases = AliasMap()
    aliases["CSR64"] = csr64
    aliases["CSC64"] = csc64
    aliases["CV64"] = cv64
    aliases["map1d"] = AffineMap("(d0)[s0, s1] -> (d0 * s1 + s0)")
    return aliases


def test_ir_builder_convert_layout_wrapper(engine: MlirJitEngine, aliases: AliasMap):
    ir_builder = MLIRFunctionBuilder(
        "convert_layout_wrapper",
        input_types=["tensor<?x?xf64, #CSR64>"],
        return_types=("tensor<?x?xf64, #CSC64>",),
        aliases=aliases,
    )
    (input_var,) = ir_builder.inputs
    convert_layout_result = ir_builder.graphblas.convert_layout(
        input_var, "tensor<?x?xf64, #CSC64>"
    )
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
    assert input_tensor.verify()

    dense_input_tensor = np.zeros([8, 8], dtype=np.float64)
    dense_input_tensor[1, 2] = 1.2
    dense_input_tensor[4, 3] = 4.3
    assert np.isclose(dense_input_tensor, input_tensor.toarray()).all()
    output_tensor = convert_layout_wrapper_callable(input_tensor)
    assert output_tensor.verify()

    assert np.isclose(dense_input_tensor, output_tensor.toarray()).all()


def test_builder_attribute(engine: MlirJitEngine, aliases: AliasMap):
    ir_builder = MLIRFunctionBuilder(
        "no_op",
        input_types=["tensor<?x?xf64, #CSR64>"],
        return_types=("tensor<?x?xf64, #CSR64>",),
        aliases=aliases,
    )
    (input_var,) = ir_builder.inputs
    ir_builder.return_vars(input_var)

    no_op = ir_builder.compile(engine=engine, passes=GRAPHBLAS_PASSES)

    assert no_op.builder == ir_builder


def test_ir_builder_triangle_count():
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
    assert input_tensor.verify()

    assert 5 == triangle_count(input_tensor)

    return


def test_ir_builder_for_loop_float_iter(engine: MlirJitEngine, aliases: AliasMap):
    # Build Function

    ir_builder = MLIRFunctionBuilder(
        "times_three", input_types=["f64"], return_types=["f64"], aliases=aliases
    )
    (input_var,) = ir_builder.inputs
    zero_f64 = ir_builder.arith.constant(0.0, "f64")
    total = ir_builder.new_var("f64")

    with ir_builder.for_loop(0, 3, iter_vars=[(total, zero_f64)]) as for_vars:
        updated_sum = ir_builder.arith.addf(input_var, total)
        for_vars.yield_vars(updated_sum)

    result_var = for_vars.returned_variable[0]
    ir_builder.return_vars(result_var)

    assert ir_builder.get_mlir()

    # Test Compiled Function
    times_three = ir_builder.compile(engine=engine)
    assert np.isclose(times_three(1.3), 3.9)


def test_ir_builder_for_loop_user_specified_vars(engine: MlirJitEngine):
    # Build Function

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
        "add_user_specified_vars",
        input_types=["i64"],
        return_types=["i64"],
    )
    (input_var,) = ir_builder.inputs
    total = ir_builder.new_var("i64")

    lower_index_var = ir_builder.arith.constant(lower_index, "index")
    upper_index_var = ir_builder.arith.constant(upper_index, "index")
    delta_index_var = ir_builder.arith.constant(delta_index, "index")
    lower_i64_var = ir_builder.arith.constant(lower_i64, "i64")
    delta_i64_var = ir_builder.arith.constant(delta_i64, "i64")
    iter_i64_var = ir_builder.new_var("i64")

    with ir_builder.for_loop(
        lower_index_var,
        upper_index_var,
        delta_index_var,
        iter_vars=[(iter_i64_var, lower_i64_var), (total, input_var)],
    ) as for_vars:
        assert lower_index_var == for_vars.lower_var_index
        assert upper_index_var == for_vars.upper_var_index
        assert delta_index_var == for_vars.step_var_index
        assert [iter_i64_var, total] == for_vars.iter_vars
        prod_of_index_vars_0 = ir_builder.arith.muli(
            for_vars.lower_var_index, for_vars.upper_var_index
        )
        prod_of_index_vars_1 = ir_builder.arith.muli(
            prod_of_index_vars_0, for_vars.step_var_index
        )
        prod_of_index_vars = ir_builder.arith.index_cast(prod_of_index_vars_1, "i64")
        prod_of_i64_vars = ir_builder.arith.muli(lower_i64_var, delta_i64_var)
        iter_index_i64 = ir_builder.arith.index_cast(for_vars.iter_var_index, "i64")
        prod_of_iter_vars = ir_builder.arith.muli(iter_index_i64, iter_i64_var)
        updated_sum_0 = ir_builder.arith.addi(total, prod_of_index_vars)
        updated_sum_1 = ir_builder.arith.addi(updated_sum_0, prod_of_i64_vars)
        updated_sum = ir_builder.arith.addi(updated_sum_1, prod_of_iter_vars)

        incremented_iter_i64_var = ir_builder.arith.addi(iter_i64_var, delta_i64_var)
        for_vars.yield_vars(incremented_iter_i64_var, updated_sum)

    result_var = for_vars.returned_variable[1]
    ir_builder.return_vars(result_var)

    assert (
        ir_builder.get_mlir()
    )  # this generated MLIR is easier to read than the above IR builder calls.

    # Test Compiled Function
    func = ir_builder.compile(engine=engine)

    calculated_sum = func(7)
    assert np.isclose(calculated_sum, expected_sum)


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
        assert all(e.verify() for e in sparse_weight_matrices)
        sparse_bias_matrices = [
            sparsify_matrix(matrix) for matrix in dense_bias_matrices
        ]
        assert all(e.verify() for e in sparse_bias_matrices)
        sparse_input_tensor = sparsify_matrix(dense_input_tensor)
        assert sparse_input_tensor.verify()
        sparse_result = dense_neural_network(
            sparse_weight_matrices,
            sparse_bias_matrices,
            sparse_input_tensor,
            clamp_threshold,
        )
        assert sparse_result.verify()
        dense_result = sparse_result.toarray()

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


ARGMINMAX_CASES = [
    # np.array([0], dtype=np.int32), # TODO do we care about this case?
    np.array([10, 15, 3, 11], dtype=np.int32),
    np.array([0, 0, 10, 15, 0, 0, 3, 11, 0], dtype=np.int32),
    np.array([0, 1, 0], dtype=np.int32),
    np.array([1], dtype=np.int32),
    np.array([-10, 15, 3, 11], dtype=np.int32),
    np.array([0, 0, -10, 15, 0, 0, 3, 11, 0], dtype=np.int32),
    np.array([0, -1, 0], dtype=np.int32),
    np.array([-1], dtype=np.int32),
]


@pytest.mark.parametrize("dense_input_tensor", ARGMINMAX_CASES)
def test_ir_builder_vector_argminmax(
    dense_input_tensor: np.ndarray, engine: MlirJitEngine, aliases: AliasMap
):
    # Build Function
    ir_builder = MLIRFunctionBuilder(
        "vector_arg_min_and_max",
        input_types=["tensor<?xi32, #CV64>"],
        return_types=["i64", "i64"],
        aliases=aliases,
    )
    (vec,) = ir_builder.inputs
    arg_min = ir_builder.graphblas.reduce_to_scalar(vec, "argmin")
    arg_max = ir_builder.graphblas.reduce_to_scalar(vec, "argmax")
    ir_builder.return_vars(arg_min, arg_max)
    vector_arg_min_and_max = ir_builder.compile(engine=engine, passes=GRAPHBLAS_PASSES)

    # Test Results
    input_tensor = sparsify_array(dense_input_tensor, [True])
    assert input_tensor.verify()
    res_min, res_max = vector_arg_min_and_max(input_tensor)

    minimum = np.min(dense_input_tensor)
    maximum = np.max(dense_input_tensor)

    dwimmed_dense_input_tensor = np.copy(dense_input_tensor)
    dwimmed_dense_input_tensor[dwimmed_dense_input_tensor == 0] = maximum + 1
    assert res_min == np.argmin(dwimmed_dense_input_tensor)

    dwimmed_dense_input_tensor = np.copy(dense_input_tensor)
    dwimmed_dense_input_tensor[dwimmed_dense_input_tensor == 0] = minimum - 1
    assert res_max == np.argmax(dwimmed_dense_input_tensor)


def test_ir_gt_thunk(engine: MlirJitEngine, aliases: AliasMap):
    # Build Function
    ir_builder = MLIRFunctionBuilder(
        "gt_thunk",
        input_types=["tensor<?x?xf64, #CSR64>", "f64"],
        return_types=["tensor<?x?xf64, #CSR64>"],
        aliases=aliases,
    )
    M, threshold = ir_builder.inputs
    twelve_scalar = ir_builder.arith.constant(12, "f64")
    thirty_four_scalar = ir_builder.arith.constant(34, "f64")
    M2 = ir_builder.graphblas.apply(M, "div", left=twelve_scalar)
    M3 = ir_builder.graphblas.apply(M2, "div", right=thirty_four_scalar)
    filtered = ir_builder.graphblas.select(M3, "gt", threshold)
    ir_builder.return_vars(filtered)
    gt_thunk = ir_builder.compile(engine=engine, passes=GRAPHBLAS_PASSES)

    # Test Results
    dense_input_tensor = np.array(
        [
            [1, 0, 0, 0, 0],
            [-9, 2, 3, 0, 0],
            [0, 0, 4, 0, 0],
            [0, 0, 5, 6, 0],
            [0, 0, 0, -9, 0],
        ],
        dtype=np.float64,
    )
    dense_input_tensor_mask = dense_input_tensor.astype(bool)
    input_tensor = sparsify_array(dense_input_tensor, [False, True])
    assert input_tensor.verify()

    for threshold in np.unique(dense_input_tensor):
        result = gt_thunk(input_tensor, threshold)
        assert result.verify()
        dense_result = result.toarray()

        expected_dense_result = np.copy(dense_input_tensor)
        expected_dense_result[dense_input_tensor_mask] /= 12.0
        expected_dense_result[dense_input_tensor_mask] **= -1
        expected_dense_result[dense_input_tensor_mask] /= 34.0
        expected_dense_result[expected_dense_result <= threshold] = 0

        assert np.all(dense_result == expected_dense_result)


REDUCE_TO_VECTOR_CASES = [
    # pytest.param(
    #     "tensor<5x4x{scalar_type}, #CSR64>",
    #     "tensor<5x{scalar_type}, #CV64>",
    #     "tensor<4x{scalar_type}, #CV64>",
    #     id="csr_fixed"
    # ), # TODO make this work
    # pytest.param(
    #     "tensor<5x4x{scalar_type}, #CSC64>",
    #     "tensor<5x{scalar_type}, #CV64>",
    #     "tensor<4x{scalar_type}, #CV64>",
    #     id="csc_fixed"
    # ), # TODO make this work
    pytest.param(
        "tensor<?x?x{scalar_type}, #CSR64>",
        "tensor<?x{scalar_type}, #CV64>",
        "tensor<?x{scalar_type}, #CV64>",
        id="csr_arbitrary",
    ),
    pytest.param(
        "tensor<?x?x{scalar_type}, #CSC64>",
        "tensor<?x{scalar_type}, #CV64>",
        "tensor<?x{scalar_type}, #CV64>",
        id="csc_arbitrary",
    ),
]


@pytest.mark.parametrize(
    "input_type_template, reduce_rows_output_type_template, reduce_columns_output_type_template",
    REDUCE_TO_VECTOR_CASES,
)
@pytest.mark.parametrize("mlir_type", ["f64"])  # TODO make this work for other types
def test_ir_reduce_to_vector(
    input_type_template: str,
    reduce_rows_output_type_template: str,
    reduce_columns_output_type_template: str,
    mlir_type: str,
    engine: MlirJitEngine,
    aliases: AliasMap,
):
    input_type = input_type_template.format(scalar_type=mlir_type)
    reduce_rows_output_type = reduce_rows_output_type_template.format(
        scalar_type=mlir_type
    )
    reduce_columns_output_type = reduce_columns_output_type_template.format(
        scalar_type=mlir_type
    )
    np_type = MLIR_TYPE_TO_NP_TYPE[mlir_type]

    # Build Functions
    ir_builder = MLIRFunctionBuilder(
        f"reduce_func_{mlir_type}",
        input_types=[input_type],
        return_types=[
            reduce_rows_output_type,
            "tensor<?xi64, #CV64>",
            reduce_rows_output_type,
            "tensor<?xi64, #CV64>",
            "tensor<?xi64, #CV64>",
            "tensor<?xi64, #CV64>",
        ],
        aliases=aliases,
    )
    (matrix,) = ir_builder.inputs

    reduced_rows = ir_builder.graphblas.reduce_to_vector(matrix, "plus", 1)
    reduced_columns = ir_builder.graphblas.reduce_to_vector(matrix, "count", 0)

    zero_scalar = ir_builder.arith.constant(0, mlir_type)
    reduced_rows_clamped = ir_builder.graphblas.apply(
        reduced_rows, "min", right=zero_scalar
    )
    reduced_rows_clamped = ir_builder.graphblas.apply(reduced_rows_clamped, "identity")

    reduced_columns_abs = ir_builder.graphblas.apply(reduced_columns, "abs")
    reduced_columns_abs = ir_builder.graphblas.apply(reduced_columns_abs, "identity")
    reduced_columns_negative_abs = ir_builder.graphblas.apply(reduced_columns, "ainv")
    reduced_columns_negative_abs = ir_builder.graphblas.apply(
        reduced_columns_negative_abs, "identity"
    )

    reduced_rows_argmin = ir_builder.graphblas.reduce_to_vector(matrix, "argmin", 1)
    reduced_columns_argmax = ir_builder.graphblas.reduce_to_vector(matrix, "argmax", 0)

    ir_builder.return_vars(
        reduced_rows,
        reduced_columns,
        reduced_rows_clamped,
        reduced_columns_negative_abs,
        reduced_rows_argmin,
        reduced_columns_argmax,
    )
    reduce_func = ir_builder.compile(engine=engine, passes=GRAPHBLAS_PASSES)

    # Test Results
    dense_input_tensor = np.array(
        [
            [1, 0, 0, 0],
            [-2, 0, 3, -4],
            [0, 0, 0, 0],
            [0, 0, 5, -6],
            [0, -7, 0, 8],
        ],
        dtype=np_type,
    )
    input_tensor = sparsify_array(dense_input_tensor, [False, True])
    input_type_is_csc = [1, 0] == SparseTensorType.parse(
        input_type, aliases
    ).encoding.ordering
    if input_type_is_csc:
        input_tensor = engine.csr_to_csc(input_tensor)

    (
        reduced_rows,
        reduced_columns,
        reduced_rows_clamped,
        reduced_columns_negative_abs,
        reduced_rows_argmin,
        reduced_columns_argmax,
    ) = reduce_func(input_tensor)

    assert reduced_rows.verify()
    assert reduced_columns.verify()
    assert reduced_rows_clamped.verify()
    assert reduced_columns_negative_abs.verify()
    assert reduced_rows_argmin.verify()
    assert reduced_columns_argmax.verify()

    reduced_rows = reduced_rows.toarray()
    reduced_columns = reduced_columns.toarray()
    reduced_rows_clamped = reduced_rows_clamped.toarray()
    reduced_columns_negative_abs = reduced_columns_negative_abs.toarray()
    reduced_rows_argmin = reduced_rows_argmin.toarray()
    reduced_columns_argmax = reduced_columns_argmax.toarray()

    expected_reduced_rows = dense_input_tensor.sum(axis=1)
    expected_reduced_columns = (
        dense_input_tensor.astype(bool).sum(axis=0).astype(np_type)
    )

    expected_reduced_rows_clamped = np.copy(expected_reduced_rows)
    expected_reduced_rows_clamped[expected_reduced_rows_clamped > 0] = 0

    expected_reduced_columns_negative_abs = -np.abs(expected_reduced_columns)

    M = dense_input_tensor.copy()
    M[dense_input_tensor == 0] = dense_input_tensor.max() + 1
    expected_reduced_rows_argmin = np.argmin(M, axis=1)

    M = dense_input_tensor.copy()
    M[dense_input_tensor == 0] = dense_input_tensor.min() - 1
    expected_reduced_columns_argmax = np.argmax(M, axis=0)

    assert np.all(reduced_rows == expected_reduced_rows)
    assert np.all(reduced_columns == expected_reduced_columns)
    assert np.all(reduced_rows_clamped == expected_reduced_rows_clamped)
    assert np.all(reduced_columns_negative_abs == expected_reduced_columns_negative_abs)
    assert np.all(reduced_rows_argmin == expected_reduced_rows_argmin)
    assert np.all(reduced_columns_argmax == expected_reduced_columns_argmax)


DIAG_CASES = [
    # pytest.param(
    #     "tensor<4x4x{scalar_type}, #CSR64>",
    #     "tensor<4x{scalar_type}, #CV64>",
    #     id="csr_fixed"
    # ), # TODO make this work
    # pytest.param(
    #     "tensor<4x4x{scalar_type}, #CSC64>",
    #     "tensor<4x{scalar_type}, #CV64>",
    #     id="csc_fixed"
    # ), # TODO make this work
    pytest.param(
        "tensor<?x?x{scalar_type}, #CSR64>",
        "tensor<?x{scalar_type}, #CV64>",
        id="csr_arbitrary",
    ),
    pytest.param(
        "tensor<?x?x{scalar_type}, #CSC64>",
        "tensor<?x{scalar_type}, #CV64>",
        id="csc_arbitrary",
    ),
]


@pytest.mark.parametrize(
    "matrix_type_template, vector_type_template",
    DIAG_CASES,
)
@pytest.mark.parametrize("mlir_type", ["f64"])  # TODO make this work for other types
def test_ir_diag(
    matrix_type_template: str,
    vector_type_template: str,
    mlir_type: str,
    engine: MlirJitEngine,
    aliases: AliasMap,
):
    matrix_type = matrix_type_template.format(scalar_type=mlir_type)
    vector_type = vector_type_template.format(scalar_type=mlir_type)
    np_type = MLIR_TYPE_TO_NP_TYPE[mlir_type]

    # Build Functions
    ir_builder = MLIRFunctionBuilder(
        f"diag_func_{mlir_type}",
        input_types=[vector_type, matrix_type],
        return_types=[
            matrix_type,
            vector_type,
        ],
        aliases=aliases,
    )
    (input_vector, input_matrix) = ir_builder.inputs

    output_matrix = ir_builder.graphblas.diag(input_vector, matrix_type)
    output_vector = ir_builder.graphblas.diag(input_matrix, vector_type)
    ir_builder.return_vars(output_matrix, output_vector)
    diag_func = ir_builder.compile(engine=engine, passes=GRAPHBLAS_PASSES)

    # Test Results
    dense_input_vector = np.array(
        [0, 0, 0, 1, 0, -2, 0, 0],
        dtype=np_type,
    )
    input_vector = sparsify_array(dense_input_vector, [True])
    assert input_vector.verify()
    dense_input_matrix = np.array(
        [
            [0, 7, 7, 0, 7],
            [0, 1, 7, 0, 0],
            [0, 1, 0, 7, 0],
            [0, 7, 0, 2, 0],
            [7, 7, 0, 0, 0],
        ],
        dtype=np_type,
    )
    input_matrix = sparsify_array(dense_input_matrix, [False, True])
    assert input_matrix.verify()
    matrix_type_is_csc = [1, 0] == SparseTensorType.parse(
        matrix_type, aliases
    ).encoding.ordering
    if matrix_type_is_csc:
        input_matrix = engine.csr_to_csc(input_matrix)

    output_matrix, output_vector = diag_func(input_vector, input_matrix)
    assert output_matrix.verify()
    assert output_vector.verify()

    output_matrix = output_matrix.toarray()
    output_vector = output_vector.toarray()

    expected_output_matrix = np.diagflat(dense_input_vector)
    expected_output_vector = np.diag(dense_input_matrix)

    assert np.all(expected_output_matrix == output_matrix)
    assert np.all(expected_output_vector == output_vector)


def test_ir_select_random(engine: MlirJitEngine, aliases: AliasMap):
    # Build Function
    ir_builder = MLIRFunctionBuilder(
        "test_select_random",
        input_types=["tensor<?x?xf64, #CSR64>", "i64", "i64"],
        return_types=["tensor<?x?xf64, #CSR64>"],
        aliases=aliases,
    )
    M, n, context = ir_builder.inputs
    filtered = ir_builder.graphblas.matrix_select_random(
        M, n, context, choose_n="choose_first"
    )
    ir_builder.return_vars(filtered)
    test_select_random = ir_builder.compile(engine=engine, passes=GRAPHBLAS_PASSES)

    # Test Results
    dense_input_tensor = np.array(
        [
            [1, 0, 0, 0, 0],
            [-9, 2, 3, 0, 0],
            [0, 0, 4, 1, 1],
            [0, 0, 5, 6, 0],
            [0, 0, 0, -9, 0],
        ],
        dtype=np.float64,
    )
    input_tensor = sparsify_array(dense_input_tensor, [False, True])
    assert input_tensor.verify()

    result = test_select_random(input_tensor, 2, 0xB00)
    assert result.verify()
    dense_result = result.toarray()

    # choose_first always selects the first N elements on the row
    expected_output_tensor = np.array(
        [
            [1, 0, 0, 0, 0],
            [-9, 2, 0, 0, 0],
            [0, 0, 4, 1, 0],
            [0, 0, 5, 6, 0],
            [0, 0, 0, -9, 0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_equal(expected_output_tensor, dense_result)


def test_ir_select_random_uniform(engine: MlirJitEngine, aliases: AliasMap):
    # Build Function
    ir_builder = MLIRFunctionBuilder(
        "test_select_random_uniform",
        input_types=["tensor<?x?xf64, #CSR64>", "i64", "!llvm.ptr<i8>"],
        return_types=["tensor<?x?xf64, #CSR64>"],
        aliases=aliases,
    )
    M, n, context = ir_builder.inputs
    filtered = ir_builder.graphblas.matrix_select_random(
        M, n, context, choose_n="choose_uniform"
    )
    ir_builder.return_vars(filtered)
    test_select_random_uniform = ir_builder.compile(
        engine=engine, passes=GRAPHBLAS_PASSES
    )

    # Test Results
    dense_input_tensor = np.array(
        [
            [1, 0, 0, 0, 0],
            [-9, 2, 3, 0, 0],
            [0, 0, 4, 1, 1],
            [0, 0, 5, 6, 0],
            [0, 0, 0, -9, 0],
        ],
        dtype=np.float64,
    )
    input_tensor = sparsify_array(dense_input_tensor, [False, True])
    assert input_tensor.verify()

    rng = ChooseUniformContext(seed=2)
    result = test_select_random_uniform(input_tensor, 2, rng)
    assert result.verify()
    dense_result = result.toarray()

    expected_row_count = np.minimum((dense_input_tensor != 0).sum(axis=1), 2)
    actual_row_count = (dense_result != 0).sum(axis=1)
    np.testing.assert_equal(expected_row_count, actual_row_count)

    # check for correct truncation
    assert len(result.indices[1]) == result.pointers[1][-1]
    assert len(result.values) == result.pointers[1][-1]


def test_ir_select_random_weighted(engine: MlirJitEngine, aliases: AliasMap):
    # Build Function
    ir_builder = MLIRFunctionBuilder(
        "test_select_random_weighted",
        input_types=["tensor<?x?xf64, #CSR64>", "i64", "!llvm.ptr<i8>"],
        return_types=["tensor<?x?xf64, #CSR64>"],
        aliases=aliases,
    )
    M, n, context = ir_builder.inputs
    filtered = ir_builder.graphblas.matrix_select_random(
        M, n, context, choose_n="choose_weighted"
    )
    ir_builder.return_vars(filtered)
    test_select_random_weighted = ir_builder.compile(
        engine=engine, passes=GRAPHBLAS_PASSES
    )

    # Test Results
    # for weighted sampling to make sense, weights must all be >= 0
    dense_input_tensor = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 2, 4, 0, 0],  # using this row for stats check below
            [0, 0, 1, 100, 1],
            [0, 0, 5, 6, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=np.float64,
    )
    input_tensor = sparsify_array(dense_input_tensor, [False, True])
    assert input_tensor.verify()

    # basic checks
    rng = ChooseWeightedContext(seed=2)
    result = test_select_random_weighted(input_tensor, 2, rng)
    assert result.verify()
    dense_result = result.toarray()

    expected_row_count = np.minimum((dense_input_tensor != 0).sum(axis=1), 2)
    actual_row_count = (dense_result != 0).sum(axis=1)
    np.testing.assert_equal(expected_row_count, actual_row_count)

    # rough statistical check of row 1
    counts = defaultdict(lambda: 0)
    n = 100
    for i in range(n):
        result = test_select_random_weighted(input_tensor, 1, rng)
        assert result.verify()
        dense_result = result.toarray()
        choice = np.argmax(dense_result[1])
        counts[choice] += 1

    assert sorted(counts.keys()) == [0, 1, 2]
    row_1 = dense_input_tensor[1]
    row_1_sum = row_1.sum()
    print(counts)
    for key, actual_count in counts.items():
        prob = row_1[key] / row_1_sum
        expected_count = prob * n
        # binomial standard deviation
        stddev = (n * prob * (1 - prob)) ** 0.5
        assert abs(expected_count - actual_count) < (
            2 * stddev
        ), f"key: {key}, expected: {expected_count}, actual: {actual_count}, stdev: {stddev}"


def test_ir_transpose(
    engine: MlirJitEngine,
    aliases: AliasMap,
):
    # Build Functions
    ir_builder = MLIRFunctionBuilder(
        "transpose_wrapper",
        input_types=["tensor<?x?xf64, #CSR64>"],
        return_types=["tensor<?x?xf64, #CSC64>"],
        aliases=aliases,
    )
    (input_matrix,) = ir_builder.inputs

    output_matrix = ir_builder.graphblas.transpose(
        input_matrix, "tensor<?x?xf64, #CSC64>"
    )
    ir_builder.return_vars(output_matrix)
    transpose_wrapper = ir_builder.compile(engine=engine, passes=GRAPHBLAS_PASSES)

    # Test Results
    dense_input_matrix = np.array(
        [
            [0, 7, 7, 0, 7],
            [0, 1, 7, 0, 0],
        ],
        dtype=np.float64,
    )
    input_matrix = sparsify_array(dense_input_matrix, [False, True])
    assert input_matrix.verify()

    output_matrix = transpose_wrapper(input_matrix)
    assert output_matrix.verify()

    output_matrix = output_matrix.toarray()

    expected_output_matrix = dense_input_matrix.T

    assert np.all(expected_output_matrix == output_matrix)
