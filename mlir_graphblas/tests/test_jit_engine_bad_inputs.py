from .jit_engine_test_utils import STANDARD_PASSES

import mlir_graphblas
import mlir_graphblas.sparse_utils
import pytest
import numpy as np

#################
# Test Fixtures #
#################


@pytest.fixture(scope="module")
def compiled_func_and_valid_args():
    mlir_text = """
#sparseTensor = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ]
}>

func @many_inputs_constant_output(
   %arg_arbitrary_size_tensor: tensor<?x?xf32>,
   %arg_fixed_size_tensor: tensor<2x3xf32>,
   %arg_partially_fixed_size_tensor: tensor<2x?xf32>,
   %arg_sparse_tensor: tensor<?x?xf32, #sparseTensor>,
   %arg_f32: f32,
   %arg_pointer: !llvm.ptr<i64>
) -> i32 {
 %c1234 = constant 1234 : i32
 return %c1234 : i32
}
"""
    engine = mlir_graphblas.MlirJitEngine()
    assert engine.add(mlir_text, STANDARD_PASSES) == ["many_inputs_constant_output"]
    callable_func = engine.many_inputs_constant_output

    indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint64)
    values = np.array([1.2, 3.4], dtype=np.float32)
    sizes = np.array([10, 20, 30], dtype=np.uint64)
    sparsity = np.array([True, True, True], dtype=np.bool8)
    sparse_tensor = mlir_graphblas.sparse_utils.MLIRSparseTensor(
        indices, values, sizes, sparsity
    )

    valid_args = [
        np.arange(7, dtype=np.float32).reshape([7, 1]),
        np.arange(6, dtype=np.float32).reshape([2, 3]),
        np.arange(8, dtype=np.float32).reshape([2, 4]),
        sparse_tensor,
        123.456,
        (10, 20, 30),
    ]

    assert callable_func(*valid_args) == 1234

    return callable_func, valid_args


##############
# Test Cases #
##############

BAD_INPUT_TEST_CASES = [  # elements are ( error_type, error_match_string, bad_arg_index, bad_arg )
    pytest.param(
        ValueError,
        "is expected to have rank",
        0,
        np.arange(10).astype(np.float32),
        id="bad_rank_for_arbitrary_size_tensor",
    ),
    pytest.param(
        ValueError,
        "is expected to have rank",
        1,
        np.arange(10).astype(np.float32),
        id="bad_rank_for_fixed_size_tensor",
    ),
    pytest.param(
        ValueError,
        r"is expected to have size [0-9]+ in the [0-9]+th dimension but has size [0-9]+",
        2,
        np.arange(10).astype(np.float32).reshape([5, 2]),
        id="bad_size_for_partially_fixed_size_tensor",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        0,
        12.34,
        id="scalar_for_arbitrary_size_tensor",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        1,
        -1234,
        id="scalar_for_fixed_size_tensor",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        2,
        0,
        id="scalar_for_partially_fixed_size_tensor",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        0,
        "bad_input_string",
        id="string_for_arbitrary_size_tensor",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        1,
        "bad_input_string",
        id="string_for_fixed_size_tensor",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        2,
        "bad_input_string",
        id="string_for_partially_fixed_size_tensor",
    ),
    pytest.param(
        TypeError,
        "cannot be cast to",
        4,
        "bad_input_string",
        id="string_for_scalar",
    ),
    pytest.param(
        TypeError,
        "is expected to be a scalar with dtype",
        4,
        np.arange(10).astype(np.float32).reshape([5, 2]),
        id="array_for_scalar",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        3,
        99,
        id="int_for_sparse_tensor",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        3,
        12.34,
        id="float_for_sparse_tensor",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        3,
        np.arange(10).astype(np.float32).reshape([5, 2]),
        id="array_for_sparse_tensor",
    ),
    pytest.param(
        TypeError,
        "is expected to be an instance of",
        3,
        "bad_input_string",
        id="string_for_sparse_tensor",
    ),
    pytest.param(
        TypeError,
        " cannot be cast to ",
        4,
        np.float64(np.finfo(np.dtype("float32")).max ** 8),
        id="f64_for_f32_scalar",
    ),
]

for np_type in (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ulonglong,
    np.float16,
    np.float64,
    np.float128,
    np.complex64,
    np.complex128,
    np.complex256,
    np.record,
    np.bool_,
    bool,
):
    if not issubclass(np_type, (bool, np.record, np.integer)):
        error_match_string = r".*12\.3.* cannot be cast to "
        if np_type is np.bool_:
            error_match_string = r"True is expected to be a scalar with dtype "
        BAD_INPUT_TEST_CASES.append(
            pytest.param(
                TypeError,
                error_match_string,
                5,
                [np_type(12.3)],
                id=f"{np_type.__name__}_array_for_i64_array",
            )
        )
    BAD_INPUT_TEST_CASES += [
        pytest.param(
            TypeError,
            "is expected to have dtype",
            0,
            np.arange(10).astype(np_type),
            id=f"bad_type_{np_type.__name__}_for_arbitrary_size_tensor",
        ),
        pytest.param(
            TypeError,
            "is expected to have dtype",
            1,
            np.arange(10).astype(np_type),
            id=f"bad_type_{np_type.__name__}_for_fixed_size_tensor",
        ),
    ]

#########
# Tests #
#########


@pytest.mark.parametrize("error_type,match,bad_arg_index,bad_arg", BAD_INPUT_TEST_CASES)
def test_jit_engine_bad_inputs(
    compiled_func_and_valid_args, error_type, match, bad_arg_index, bad_arg
):
    compiled_func, valid_args = compiled_func_and_valid_args
    with pytest.raises(error_type, match=match):
        invalid_args = list(valid_args)
        invalid_args[bad_arg_index] = bad_arg
        compiled_func(*invalid_args)


def test_jit_engine_incorrect_number_of_inputs(compiled_func_and_valid_args):
    compiled_func, valid_args = compiled_func_and_valid_args
    with pytest.raises(ValueError, match=r"[0-9] args but got [0-9]"):
        invalid_args = valid_args + valid_args
        compiled_func(*invalid_args)

    with pytest.raises(ValueError, match=r"[0-9] args but got 0."):
        compiled_func()
