import mlir
import pytest
import numpy as np

from mlir_graphblas import MlirJitEngine
from mlir_graphblas.engine import parse_mlir_string
from mlir_graphblas.sparse_utils import MLIRSparseTensor
from mlir_graphblas.mlir_builder import MLIRVar, MLIRFunctionBuilder
from mlir_graphblas.functions import (
    Transpose,
    MatrixSelect,
    MatrixReduceToScalar,
    MatrixMultiply,
)

from typing import List


@pytest.fixture(scope="module")
def engine():
    jit_engine = MlirJitEngine()

    jit_engine.add(
        """
#trait_densify = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,
    affine_map<(i,j) -> (i,j)>
  ],
  iterator_types = ["parallel", "parallel"],
  sparse = [
    [ "D", "S" ],
    [ "D", "D" ]
  ],
  sparse_dim_map = [
    affine_map<(i,j) -> (j,i)>,
    affine_map<(i,j) -> (i,j)>
  ]
}

!SparseTensor = type !llvm.ptr<i8>

func @densify(%argA: !SparseTensor) -> tensor<8x8xf64> {
  %output_storage = constant dense<0.0> : tensor<8x8xf64>
  %arga = linalg.sparse_tensor %argA : !SparseTensor to tensor<8x8xf64>
  %0 = linalg.generic #trait_densify
    ins(%arga: tensor<8x8xf64>)
    outs(%output_storage: tensor<8x8xf64>) {
      ^bb(%A: f64, %x: f64):
        linalg.yield %A : f64
    } -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}
""",
        [
            "--test-sparsification=lower",
            "--linalg-bufferize",
            "--convert-scf-to-std",
            "--func-bufferize",
            "--tensor-bufferize",
            "--tensor-constant-bufferize",
            "--finalizing-bufferize",
            "--convert-linalg-to-loops",
            "--convert-scf-to-std",
            "--convert-std-to-llvm",
        ],
    )
    return jit_engine


def test_ir_builder_transpose_wrapper(engine: MlirJitEngine):
    # Build Function
    transpose_function = Transpose()

    input_var = MLIRVar("input_tensor", "!llvm.ptr<i8>")

    ir_builder = MLIRFunctionBuilder(
        "transpose_wrapper", input_var, return_type="!llvm.ptr<i8>"
    )
    transpose_result = ir_builder.call(transpose_function, input_var)
    ir_builder.return_var(transpose_result)

    assert ir_builder.get_mlir()

    # Test Compiled Function
    transpose_wrapper_callable = ir_builder.compile(engine=engine)

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
    assert np.isclose(dense_input_tensor, engine.densify(input_tensor)).all()

    output_tensor = transpose_wrapper_callable(input_tensor)

    assert np.isclose(
        dense_input_tensor, engine.densify(input_tensor)
    ).all(), f"{input_tensor} values unexpectedly changed."
    assert np.isclose(dense_input_tensor.T, engine.densify(output_tensor)).all()

    return


def test_ir_builder_triple_transpose(engine: MlirJitEngine):
    # Build Function

    input_var = MLIRVar("input_tensor", "!llvm.ptr<i8>")

    ir_builder = MLIRFunctionBuilder(
        "in_place_double_transpose", input_var, return_type="!llvm.ptr<i8>"
    )
    # Use different instances of Tranpose to ideally get exactly one transpose helper in the final MLIR text
    inter1 = ir_builder.call(Transpose(), input_var)
    inter2 = ir_builder.call(Transpose(), inter1)
    return_var = ir_builder.call(Transpose(), inter2)
    ir_builder.return_var(return_var)

    mlir_text = ir_builder.get_mlir(make_private=False)
    ast = parse_mlir_string(mlir_text)
    # verify there are exactly two functions
    private_func, public_func = [
        node for node in ast.body if isinstance(node, mlir.astnodes.Function)
    ]
    assert private_func.name.value == "transpose"
    assert private_func.visibility == "private"
    assert public_func.name.value == "in_place_double_transpose"
    assert public_func.visibility == "public"

    # verify in_place_triple_transpose has three transpose calls
    region = public_func.body
    (block,) = region.body
    call_op_1, call_op_2, call_op_3, return_op = block.body
    assert (
        call_op_1.op.func.value
        == call_op_2.op.func.value
        == call_op_3.op.func.value
        == "transpose"
    )
    (return_op_type,) = [
        t for t in mlir.dialects.standard.ops if t.__name__ == "ReturnOperation"
    ]
    assert isinstance(return_op.op, return_op_type)

    # Test Compiled Function
    triple_transpose_callable = ir_builder.compile(engine=engine)

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
    assert np.isclose(dense_input_tensor, engine.densify(input_tensor)).all()

    transposed_tensor = triple_transpose_callable(input_tensor)

    assert np.isclose(
        dense_input_tensor, engine.densify(input_tensor)
    ).all(), f"{input_tensor} values unexpectedly changed."
    assert np.isclose(dense_input_tensor.T, engine.densify(transposed_tensor)).all()

    return


def test_ir_builder_triangle_count(engine: MlirJitEngine):
    # Build Function
    csr_to_csc_function = Transpose(swap_sizes=False)
    matrix_select_triu_function = MatrixSelect("TRIU")
    matrix_select_tril_function = MatrixSelect("TRIL")
    matrix_reduce_function = MatrixReduceToScalar()
    mxm_plus_pair_function = MatrixMultiply("plus_pair", mask=True)

    A_var = MLIRVar("A", "!llvm.ptr<i8>")

    ir_builder = MLIRFunctionBuilder("triangle_count", A_var, return_type="f64")
    U_var = ir_builder.call(matrix_select_triu_function, A_var)
    L_var = ir_builder.call(matrix_select_tril_function, A_var)
    U_csc = ir_builder.call(csr_to_csc_function, U_var)
    C_var = ir_builder.call(mxm_plus_pair_function, L_var, U_csc, L_var)
    reduce_result = ir_builder.call(matrix_reduce_function, C_var)
    ir_builder.return_var(reduce_result)

    assert ir_builder.get_mlir()

    # Test Compiled Function
    triangle_count_internal = ir_builder.compile(engine=engine)

    def triangle_count(A: MLIRSparseTensor) -> int:
        answer = triangle_count_internal(A)
        return int(answer)

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

    assert 5 == triangle_count(input_tensor)

    return
